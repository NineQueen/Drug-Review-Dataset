"""Verify Word2Vec registry, configuration persistence and model handoff."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def main(reference_repo=None):
    import numpy as np
    from pyspark.sql import SparkSession, functions as F
    from pyspark.ml.functions import vector_to_array
    from pyspark.ml.linalg import VectorUDT
    from pyspark.ml.feature import Word2VecModel
    from src.embeddings import EmbeddingModel
    import src.embeddings.word2vec as module
    assert SparkSession.getActiveSession() is None
    assert "word2vec" in EmbeddingModel.train_registry and "word2vec" in EmbeddingModel.predict_registry
    original = module.DEFAULT_CONFIG.copy()
    for config in [{"window_szie": 3}, {"vector_size": 0}, {"max_iter": True}, {"step_size": float("nan")}, {"pooling": "minmax"}]:
        try:
            module._configuration(config)
        except ValueError:
            pass
        else:
            raise AssertionError(config)
    np.testing.assert_allclose(module._meanmax(["a","b","OOV"],{"a":np.array([-3.,-1.]),"b":np.array([-2.,-4.])},2),[-5/3,-5/3,-2,-1])
    (ROOT/"tmp").mkdir(exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix="word2vec_check_", dir=ROOT/"tmp"))
    spark = SparkSession.builder.master("local[*]").appName("Word2VecDeliveryCheck").config("spark.driver.memory","4g").config("spark.sql.parquet.columnarReaderBatchSize","512").config("spark.sql.execution.pythonUDF.arrow.enabled","false").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        train_rows=[(i,float(i%10+1),float(i%10),["pain","not","good","works"] if i%2 else ["pain","good","works"],1.) for i in range(20)]
        schema="uniqueID long, rating double, rating_p1 double, words array<string>, weight double"
        train=spark.createDataFrame(spark.sparkContext.parallelize(train_rows,2),schema)
        valid=spark.createDataFrame(spark.sparkContext.parallelize([(100,1.,0.,[],1.),(101,2.,1.,["OOV"],1.),(102,3.,2.,["pain","pain","OOV"],1.),(103,4.,3.,["validationonly"],1.)],2),schema).drop("weight")
        instance=EmbeddingModel(train,valid,valid)
        with patch.object(module,"EMBEDDING_MODELS_DIR",temp):
            # Standard runner call with no configuration must train defaults.
            EmbeddingModel.train_registry["word2vec"](instance)
            record=json.loads((temp/"word2vec/embedding_config.json").read_text())
            assert record["config"]==original and record["output_dimension"]==600
            names=[("word2vec",600), ("word2vec_custom",8)]
            custom={"vector_size":8,"window_size":3,"min_count":1,"max_iter":2,"pooling":"p01"}
            module.word2vec_train(instance,"word2vec_custom",config=custom)
            assert custom=={"vector_size":8,"window_size":3,"min_count":1,"max_iter":2,"pooling":"p01"}
            assert module.DEFAULT_CONFIG==original
            for name,dimension in names:
                model=Word2VecModel.load(str(temp/name))
                assert "validationonly" not in {r.word for r in model.getVectors().select("word").collect()}
                outputs=EmbeddingModel.predict_registry["word2vec"](instance,name)
                for source,out in zip([train,valid,valid],outputs):
                    assert out.columns==[c for c in EmbeddingModel.OUTPUT_COLUMNS if c in source.columns or c=="embedding"]
                    assert out.schema["embedding"].dataType==VectorUDT()
                    assert out.select("uniqueID","rating").orderBy("uniqueID").collect()==source.select("uniqueID","rating").orderBy("uniqueID").collect()
                    assert out.filter(F.size(vector_to_array("embedding"))!=dimension).count()==0
                assert all(np.all(r.embedding.toArray()==0) for r in outputs[1].filter("uniqueID in (100,101,103)").collect())
            with patch.dict(module.DEFAULT_CONFIG,{"pooling":"meanmax","vector_size":111}):
                out=EmbeddingModel.predict_registry["word2vec"](instance,"word2vec_custom")[1]
                assert out.first().embedding.size==8
            before=Word2VecModel.load(str(temp/"word2vec_custom")).uid
            module.word2vec_train(instance,"word2vec_custom",config={**custom,"pooling":"meanmax"})
            assert Word2VecModel.load(str(temp/"word2vec_custom")).uid!=before
            out=instance.word2vec_predict("word2vec_custom")[1]
            assert out.first().embedding.size==16
            out.write.parquet(str(temp/"roundtrip"))
            restored=spark.read.parquet(str(temp/"roundtrip"))
            assert not out.exceptAll(restored).take(1) and not restored.exceptAll(out).take(1)
        result=dict(status="passed",python=sys.version,spark=spark.version,java=spark.sparkContext._jvm.java.lang.System.getProperty("java.version"),default_dimension=600,custom_dimensions=[8,16],registry_and_narrow_columns=True,train_only_vocabulary=True,config_persistence=True,existing_name_retrained=True,empty_oov=True,negative_max=True,parquet_roundtrip=True)
        if reference_repo:
            from src.text_features import text_feature_engineering
            reference_repo=Path(reference_repo)
            frames=tuple(spark.read.parquet(str(reference_repo/"data/processed/split"/f"{s}.parquet")).coalesce(2).limit(80).cache() for s in ["train","valid","test"])
            real=EmbeddingModel(*frames)
            outputs=real.word2vec_predict()
            errors=[]
            for split,frame,out in zip(["train","valid","test"],frames,outputs):
                baseline=spark.read.parquet(str(reference_repo/"data/processed/embeddings/w2v_t20261108_225759_d300_meanmax"/f"{split}.parquet"))
                pairs=out.select("uniqueID",vector_to_array("embedding").alias("new")).join(baseline.select("uniqueID",vector_to_array("embedding").alias("old")),"uniqueID")
                assert pairs.count()==frame.count()
                error=pairs.select(F.array_max(F.zip_with("new","old",lambda x,y:F.abs(x-y))).alias("e")).agg(F.max("e")).first()[0]
                assert error<1e-7;errors.append(error)
            sample=frames[0].select("uniqueID","review","words")
            latest=text_feature_engineering(sample.select("uniqueID","review"),True).select("uniqueID",F.col("words").alias("latest_words"))
            assert not sample.join(latest,"uniqueID").filter(~F.col("words").eqNullSafe(F.col("latest_words"))).take(1)
            result.update(historical_comparison_rows=240,max_vector_error=max(errors),latest_tokenizer_sample_parity=True)
        (ROOT/"tmp/word2vec_delivery_validation.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print("WORD2VEC_DELIVERY_PASSED",json.dumps(result),flush=True)
    finally:
        spark.stop()


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--reference-repo");args=parser.parse_args();main(args.reference_repo)
