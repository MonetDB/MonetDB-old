create system procedure lidarattach(fname string) external name lidar.attach;
create system procedure lidarload(tname string) external name lidar.load;
create system procedure lidarexport(tname string, fname string, format string) external name lidar.export;
