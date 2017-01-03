create procedure lidarattach(fname string) external name lidar.attach;
create procedure lidarattach(fname string, tname string) external name lidar.attach;
create procedure lidarattach(fname string, tname string, params string) external name lidar.attach;
create procedure lidarload(tname string) external name lidar.load;
create procedure lidarexport(tname string, fname string, format string) external name lidar.export;
create procedure lidarunload(tname string) external name lidar.unload;
create procedure lidarchecktab(tname string) external name lidar.checktab;
create procedure lidaranalyzetab(status int, tname string) external name lidar.analyzetab;
