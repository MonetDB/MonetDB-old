

create system procedure listdir(dirname string) external name fits.listdir;
create system procedure fitsattach(fname string) external name fits.attach;
create system procedure fitsload(tname string) external name fits.load;
create system procedure listdirpat(dirname string,pat string) external name fits.listdirpattern;
