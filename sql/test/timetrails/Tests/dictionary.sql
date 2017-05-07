--Tests for meta information extraction by  Grafana

SELECT * FROM rooms;

SELECT * FROM timetrails.metrics();

SELECT * FROM timetrails.tags('rooms');

SELECT * FROM timetrails.fields('rooms');

SELECT * FROM timetrails.getlayout('rooms');
