package localio;
import java.io.*;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.fs.permission.FsPermission;
public class WindowsLocalFileSystem extends RawLocalFileSystem {
 @Override public boolean mkdirs(Path p, FsPermission permission) throws IOException {
  File f=pathToFile(p); return f.isDirectory() || f.mkdirs();
 }
 @Override public void setPermission(Path p, FsPermission permission) throws IOException { }
 @Override public FileStatus getFileStatus(Path p) throws IOException {
  File f=pathToFile(p);
  if(!f.exists()) throw new FileNotFoundException(p.toString());
  return new FileStatus(f.length(), f.isDirectory(), 1, 4096, f.lastModified(),
                        p.makeQualified(getUri(),getWorkingDirectory()));
 }
 @Override public FileStatus[] listStatus(Path p) throws IOException {
  File f=pathToFile(p);
  if(!f.isDirectory()) return new FileStatus[]{getFileStatus(p)};
  File[] files=f.listFiles();
  if(files==null) throw new IOException("Cannot list "+p);
  FileStatus[] result=new FileStatus[files.length];
  for(int i=0;i<files.length;i++) result[i]=getFileStatus(new Path(files[i].toURI()));
  return result;
 }
}