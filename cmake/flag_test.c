#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **av) {
	char buf[24];
	double d = atof("4.2");
	if (argc == 0 || strchr(av[0], av[0][0]) != NULL)
		snprintf(buf, sizeof(buf), "%f", d);
	return 0;
}
