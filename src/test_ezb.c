/*
 * EZ Bootstrap (EZB) - Test runner
 * 
 * Runs all tests for EZB implementations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

int main(int argc, char *argv[]) {
    int failures = 0;
    
    printf("EZ Bootstrap (EZB) - Test Suite\n");
    printf("================================\n\n");
    
    /* Test single condition */
    printf("Testing EZB Single...\n");
    printf("---------------------\n");
    
    int status = system("./ezb_single --test");
    if (status != 0) {
        failures++;
    }
    
    printf("\n");
    
    /* Test design matrix */
    printf("Testing EZB Design Matrix...\n");
    printf("----------------------------\n");
    
    status = system("./ezb_design_matrix --test");
    if (status != 0) {
        failures++;
    }
    
    printf("\n");
    printf("================================\n");
    if (failures == 0) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("%d test suite(s) failed\n", failures);
        return 1;
    }
}

