// From https://github.com/Framstag/libosmscout/pull/251/files#diff-40f8e3cdfc36152528d5235258332447
#include <stdio.h>

int main()
{
	__int64 off=0;

	_fseeki64(NULL, off, SEEK_SET);

	return 0;
}
