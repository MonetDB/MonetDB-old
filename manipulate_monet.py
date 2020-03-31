from pymonetdb import mapi
import subprocess
import sys
import os

available_cases = [ "start", "stop", "cleanup", "create_tables", "execute" ]
all_servers = [ 50000, 50001, 50002 ]
all_db_paths = [ "/tmp/mdb1", "/tmp/mdb2", "/tmp/mdb3" ]
hostname = "localhost"
username = "monetdb"
password = "monetdb"

if len(sys.argv) < 2:
    print("Usage: python <script.py> <start | stop | cleanup | execute>")
    sys.exit()
else:
    monetcase = sys.argv[1]
    if monetcase not in available_cases:
        print("Usage: python <script.py> <start | stop | cleanup | execute>")
        sys.exit()



def monet_start():
    for i in range(0, len(all_servers)):
        print(f"Starting mserver on port: {all_servers[i]} and dbpath: {all_db_paths[i]}")
        p = subprocess.Popen([f"mserver5 --dbpath={all_db_paths[i]} --set mapi_port={all_servers[i]} --set monet_daemon=yes"], shell=True)


def monet_stop():
    print(f"Stoping {len(all_servers)} mserver5 instances")
    p = subprocess.Popen(["t=`pgrep mserver5` && kill -9 $t"], shell=True)
    p.communicate()


def monet_cleanup():
    for db_path in all_db_paths:
        print(f"Cleaning directory {db_path}")
        p = subprocess.Popen([f"rm -rf {db_path}"], shell=True)
        p.communicate()


# Each of the 3 servers holds a table s1 with the same structure (replicated schema) 
# A query select * from s1 comes to one of the servers. The hitchhiker has to find all
# the databases that s1 exists and put the necessary move statements. In this case it 
# needs to put 3 different statements (since there are 3 mservers holding the same table)
# P.S: In this query we are not using merge tables!
def monet_execute():
    server = mapi.Connection()
    for i in range(0, len(all_servers)):
        database = all_db_paths[i].split("/")[2]
        server.connect(hostname=hostname, port=all_servers[i], 
                       username=username, password=password,
                       database=database, language="sql")
        if monetcase == "create_tables":
            server.cmd(f"sCREATE TABLE s1(i INT);")
            server.cmd(f"sINSERT INTO s1 VALUES ({i + 23}), ({i + 42});")
        elif monetcase == "execute":
            # we need to execute the query only on 1 mserver5 
            # the hitchhiker will do the rest of the work
            server.cmd("sexplain SELECT * FROM s1;")
            break
        


def main():
    if monetcase == "start":
        monet_start()
    elif monetcase == "stop":
        monet_stop()
    elif monetcase == "cleanup":
        monet_cleanup()
    else:
        monet_execute()


if __name__ == "__main__":
    main()
