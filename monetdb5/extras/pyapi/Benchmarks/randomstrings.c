

// Small file for generating random strings of the same length
// Parameters are string_size (MB), string_length (#chars), file name
// The strings are saved in the specified file separated by spaces


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


#define TYPE_INT 1
#define TYPE_STRING 2

int main(int argc, char *argv[])
{
	srand(377);

	int type;

	if (argc != 4) {
		fprintf(stderr, "Invalid number of arguments for random generation, expected \"size[mb] string_length[chars] result_file[name]\"\n");
		exit(1);
	}
	type = TYPE_STRING;
	if (argv[2][0] == 'i') {
		type = TYPE_INT;
	}

	if (type == TYPE_STRING) {
		double size_mb;
		size_t string_count;
		size_t string_length;

		size_mb = strtod(argv[1], NULL);
		string_length = atoi(argv[2]);

		string_count = (size_t) (size_mb * 1000 * 1000) / string_length;

		char option_string[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
		size_t options = strlen(option_string);
		const size_t rand_division = RAND_MAX / options;
		size_t result_size = string_length * string_count + string_count - 1;
		char *result = malloc(sizeof(char) * result_size + 1);
		result[result_size] = '\0';
		char sep = '\n';
		size_t i, j;

		for(i = 0; i < result_size; i += string_length + 1) {
			for(j = 0; j < string_length; j++) {
				result[i + j] = option_string[rand() / rand_division];
			}
		}
		for(i = string_length; i < result_size; i += string_length + 1) {
			result[i] = sep;
		}
		size_t MAX_BUFFER_SIZE = 100000;
		{
			FILE *f = fopen(argv[3], "w");
			for(i = 0; i < result_size / MAX_BUFFER_SIZE; i++) {
				size_t tempindex = (i + 1) * MAX_BUFFER_SIZE;
				char tmp = result[tempindex];
				result[tempindex] = '\0';
				fprintf(f, "%s", result + i * MAX_BUFFER_SIZE);
				result[tempindex] = tmp;
			}
			fprintf(f, "%s", result + i * MAX_BUFFER_SIZE);
			fprintf(f, "\n");
			fclose(f);
		}

		free(result);
	} else {
		double size_mb;
		size_t integer_count;
		size_t i;

		size_mb = strtod(argv[1], NULL);
		double min_int = -2147483640;
		double max_int = 2147483640;
		integer_count = size_mb * 1000 * 1000 / 4;


		FILE *f = fopen(argv[3], "w");
		for(i = 0; i < integer_count; i++) {
			fprintf(f, "%d\n",(int)(((double)rand() / (double)RAND_MAX) * (max_int - min_int) + min_int));
		}
		fclose(f);

	}
	return 0;
}
