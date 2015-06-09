
#include "unicode.h"

#include <string.h>

int utf8_strlen(const char *utf8_str)
{
    int utf8_char_count = 0;
    int char_count = strlen(utf8_str);
    int i;
    //we traverse the string and simply count the amount of utf8 characters in the string
    for(i = 0; i < char_count; )
    {
        int offset = utf8_length(utf8_str[i]);
        if (offset < 0) return -1; //invalid utf8 character
        i += offset;
        utf8_char_count++;
    }
    return utf8_char_count;
}

int utf32_strlen(const uint32_t *utf32_str)
{
	size_t i = 0;
	while(utf32_str[i] != 0)
		i++;
	return (i - 1);
}

int utf8_length(unsigned char utf8_char)
{
    //the first byte tells us how many bytes the utf8 character uses
    if      (utf8_char < 0b10000000) return 1;
    else if (utf8_char < 0b11100000) return 2;
    else if (utf8_char < 0b11110000) return 3;
    else if (utf8_char < 0b11111000) return 4;
    else return -1; //invalid utf8 character, the maximum value of the first byte is 0b11110111
}

int utf32_char_to_utf8_char(size_t position, char *utf8_storage, uint32_t utf32_char)
{
    int utf8_size = 4;
    if      (utf32_char < 0x80)        utf8_size = 1;
    else if (utf32_char < 0x800)       utf8_size = 2;
    else if (utf32_char < 0x10000)     utf8_size = 3;
    else if (utf32_char > 0x0010FFFF)  return -1; //utf32 character is out of legal range

    switch(utf8_size)
    {
        case 4:
            utf8_storage[position + 3] = ((utf32_char | 0b10000000) & 0b10111111); utf32_char >>= 6;
            utf8_storage[position + 2] = ((utf32_char | 0b10000000) & 0b10111111); utf32_char >>= 6;
            utf8_storage[position + 1] = ((utf32_char | 0b10000000) & 0b10111111); utf32_char >>= 6;
            utf8_storage[position]     =  (utf32_char | 0b11110000);
            return utf8_size;
        case 3:
            utf8_storage[position + 2] = ((utf32_char | 0b10000000) & 0b10111111); utf32_char >>= 6;
            utf8_storage[position + 1] = ((utf32_char | 0b10000000) & 0b10111111); utf32_char >>= 6;
            utf8_storage[position]     =  (utf32_char | 0b11100000);
            return utf8_size;
        case 2:
            utf8_storage[position + 1] = ((utf32_char | 0b10000000) & 0b10111111); utf32_char >>= 6;
            utf8_storage[position]     =  (utf32_char | 0b11000000);
            return utf8_size;
        default:
            utf8_storage[position]     = utf32_char;
            return utf8_size;
    }
}

bool utf32_to_utf8(size_t offset, size_t size, char *utf8_storage, const uint32_t *utf32)
{
    size_t i = 0;
    int position = 0;
    int shift;
    for(i = 0; i < size; i++)
    {
        if (utf32[offset + i] == 0) 
        {
            utf8_storage[position] = '\0';
            return true;
        }

        shift = utf32_char_to_utf8_char(position, utf8_storage, utf32[offset + i]);
        if (shift < 0) return false;
        position += shift;
    }
    return true;
}


int utf8_char_to_utf32_char(size_t position, uint32_t *utf32_storage, int offset, const unsigned char *utf8_char)
{
    unsigned char bytes[4];
    int utf8_size = 4;
    bytes[0] = utf8_char[offset]; bytes[1] = 0xFF; bytes[2] = 0xFF; bytes[3] = 0xFF;
    //the first byte tells us how many bytes the utf8 character uses
    if      (bytes[0] < 0b10000000) utf8_size = 1;
    else if (bytes[0] < 0b11100000) utf8_size = 2;
    else if (bytes[0] < 0b11110000) utf8_size = 3;
    else if (bytes[0] < 0b11111000) utf8_size = 4;
    else return -1; //invalid utf8 character, the maximum value of the first byte is 0b11110111

    switch(utf8_size)
    {
        case 4: 
            bytes[3] = utf8_char[offset + 3];
            if (bytes[3] > 0b11000000) return -1; //invalid utf8 character, the maximum value of the second, third and fourth bytes is 0b10111111
        case 3: 
            bytes[2] = utf8_char[offset + 2];
            if (bytes[2] > 0b11000000) return -1;
        case 2: 
            bytes[1] = utf8_char[offset + 1];
            if (bytes[1] > 0b11000000) return -1;
    }

    utf32_storage[position] = 0;

    switch(utf8_size)
    {
        case 4:
            utf32_storage[position] |= (0b00111111 & bytes[3]);
            utf32_storage[position] |= (0b00111111 & bytes[2]) << 6;
            utf32_storage[position] |= (0b00111111 & bytes[1]) << 12;
            utf32_storage[position] |= (0b00000111 & bytes[0]) << 18;
            return utf8_size;
        case 3:
            utf32_storage[position] |= (0b00111111 & bytes[2]);
            utf32_storage[position] |= (0b00111111 & bytes[1]) << 6;
            utf32_storage[position] |= (0b00001111 & bytes[0]) << 12;
            return utf8_size;
        case 2:
            utf32_storage[position] |= (0b00111111 & bytes[1]);
            utf32_storage[position] |= (0b00011111 & bytes[0]) << 6;
            return utf8_size;
        default:
            utf32_storage[position] |= 0b01111111 & bytes[0];
            return utf8_size;
    }
}

bool utf8_to_utf32(size_t offset, size_t size, uint32_t *utf32_storage, const unsigned char *utf8)
{
    size_t i = 0;
    int position = 0;
    int shift;
    for(i = 0; i < size; i++)
    {
        if (utf8[offset + position] == 0) 
        {
            utf32_storage[i] = '\0';
            return true;
        }

        shift = utf8_char_to_utf32_char(i, utf32_storage, offset + position, utf8);
        if (shift < 0) return false;
        position += shift;
    }
    return true;
}
