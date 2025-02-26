#include "stdio.h"

extern "C"
void my_test_start(int num) {
    printf("my_test_start %d\n", num);
}

extern "C"
void my_test_end(int num) {
    printf("my_test_end %d\n", num);
}
