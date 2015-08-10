

// Small file for generating random strings of the same length
// Parameters are string_size (MB), string_length (#chars), file name
// The strings are saved in the specified file separated by spaces


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


int main(int argc, char *argv[])
{
	srand(377);

	if (argc != 4) {
		fprintf(stderr, "Invalid number of arguments for random generation, expected \"size[mb] string_length[chars] result_file[name]\"\n");
		exit(1);
	}

	double size_mb;
	int string_count;
	int string_length;

	size_mb = strtod(argv[1], NULL);
	string_length = atoi(argv[2]);

	string_count = (int) (size_mb * 1000 * 1000) / string_length;

	char option_string[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
	size_t options = strlen(option_string);
	const size_t rand_division = RAND_MAX / options;
	size_t result_size = string_length * string_count + string_count - 1;
	char *result = malloc(sizeof(char) * result_size + 1);
	result[result_size] = '\0';
	char sep = ' ';
	size_t i, j;

	for(i = 0; i < result_size; i += string_length + 1) {
		for(j = 0; j < string_length; j++) {
			result[i + j] = option_string[rand() / rand_division];
		}
	}
	for(i = string_length; i < result_size; i += string_length + 1) {
		result[i] = sep;
	}
	int MAX_BUFFER_SIZE = 100000;
	{
		FILE *f = fopen(argv[3], "w");
		for(i = 0; i < result_size / MAX_BUFFER_SIZE; i++) {
			int tempindex = (i + 1) * MAX_BUFFER_SIZE;
			char tmp = result[tempindex];
			result[tempindex] = '\0';
			fprintf(f, "%s", result + i * MAX_BUFFER_SIZE);
			result[tempindex] = tmp;
		}
		fprintf(f, "%s", result + i * MAX_BUFFER_SIZE);
		fclose(f);
	}

	free(result);
	return 0;
}
