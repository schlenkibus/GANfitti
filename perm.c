#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
static const int alphabetSize = sizeof(alphabet) - 1;

static void bruteImpl(char* str, size_t index, size_t maxDepth)
{
    for (size_t i = 0; i < alphabetSize; ++i)
    {
        str[index] = alphabet[i];

        if (index == maxDepth - 1) printf("%s\n", str);
        else bruteImpl(str, index + 1, maxDepth);
    }
}

void bruteSequential(size_t maxLen)
{
    char* buf = calloc(maxLen + 1, sizeof(char));

    for (size_t len = 1; len <= maxLen; ++len)
    {
        bruteImpl(buf, 0, len);
    }

    free(buf);
}

int main(int argc, char** argv)
{
    bruteSequential(6);
}
