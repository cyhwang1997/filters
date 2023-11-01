/*
 * ============================================================================
 *
 *       Filename:  main.cc
 *
 *         Author:  Prashant Pandey (), ppandey2@cs.cmu.edu
 *   Organization:  Carnegie Mellon University
 *
 * ============================================================================
 */

#include <iostream>
#include <fstream>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h>  // portable to all x86 compilers
#include <tmmintrin.h>
#include <openssl/rand.h>
#include <sys/time.h>

#include <set>
#include <time.h>

#include "vqf_filter.h"

#include <cstring>
#include <vector>

// FOR ZIPFIAN
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <random>
#include "zipfian_int_distribution.h"

#include "utils.h"
#include "zipfian_generator.h"
#include "scrambled_zipfian_generator.h"

#define TEST_NUM 1

// MD5
#include <openssl/md5.h>

using namespace ycsbc;
using namespace std;

#ifdef __AVX512BW__
extern __m512i SHUFFLE [];
extern __m512i SHUFFLE_REMOVE [];
extern __m512i SHUFFLE16 [];
extern __m512i SHUFFLE_REMOVE16 [];
#endif

uint64_t tv2usec(struct timeval *tv) {
  return 1000000 * tv->tv_sec + tv->tv_usec;
}

/* Print elapsed time using the start and end timeval */
void print_time_elapsed(const char* desc, struct timeval* start, struct
      timeval* end, uint64_t ops, const char *opname) {
  uint64_t elapsed_usecs = tv2usec(end) - tv2usec(start);
  printf("%s Total Time Elapsed: %f seconds", desc, 1.0*elapsed_usecs / 1000000);
  if (ops) {
    printf(" (%f nanoseconds/%s)", 1000.0 * elapsed_usecs / ops, opname);
    printf(" (Throughput %f Mops/sec]", 1.0 * ops / elapsed_usecs);
  }
  printf("\n");
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "Please specify three arguments: \n \
                     1. Log2 of the number of slots in the CQF.\n \
                     2. Load factor (0 - 95).\n \
                     3. Zipfian constant (double).\n");
    exit(1);
  }
#ifdef __AVX512BW__
  printf("[CYDBG] AVX512 is used\n");
#else
  printf("[CYDBG] AVX2 is used\n");
#endif

  uint64_t qbits = atoi(argv[1]);
  uint64_t nslots = (1ULL << qbits);
  uint64_t load_factor = atoi(argv[2]);
  uint64_t nvals = load_factor*nslots/100;
  uint64_t *vals;
  uint64_t *other_vals;
  double zipf_const = std::stod(argv[3]);

//  uint64_t sizeval = atoi(argv[4]);
//  double sizevalR = 1.0 * sizeval / 100;

  double insertion_throughput = 0.0;
  double positive_throughput = 0.0;
  double negative_throughput = 0.0;
  double remove_throughput = 0.0;

  vector<uint64_t> uniq_vals(nvals, 0);

  /* Repeat the test for TEST_NUM times. */
  for (int test_num = 0; test_num < TEST_NUM; test_num++) {
    vqf_filter *filter;	

    /* initialize vqf filter */
    if ((filter = vqf_init(nslots)) == NULL) {
      fprintf(stderr, "Can't allocate vqf filter.");
      exit(EXIT_FAILURE);
    }

    if (zipf_const == 0) {
      // Generate random values
      vals = (uint64_t*)malloc(nvals*sizeof(vals[0]));
      mt19937 rng(42);
      uniform_int_distribution<uint64_t> dist(0, filter->metadata.range - 1);
//      RAND_bytes((unsigned char *)vals, sizeof(*vals) * nvals);
      for (uint64_t i = 0; i < nvals; i++) {
        vals[i] = dist(rng);
//        vals[i] = (1 * vals[i]) % filter->metadata.range;i
//        uniq_vals[i] = vals[i];
      }
    } else {
      Generator<uint64_t> *key_chooser_;
      key_chooser_ = new ScrambledZipfianGenerator(0, filter->metadata.range, zipf_const);
      vals = (uint64_t*)malloc(nvals*sizeof(vals[0]));
      for (uint64_t i = 0; i < nvals; i++) {
        vals[i] = key_chooser_->Next() % filter->metadata.range;
        uniq_vals[i] = vals[i];
      }
      printf("Zipfian Created\n");
/*      uint64_t *foo;
      foo = (uint64_t*)malloc((filter->metadata.range)*sizeof(foo[0]));
      printf("[CYDBG] %ld\n", filter->metadata.range);
      for (uint64_t i = 0; i < filter->metadata.range; i++) {
        foo[i] = i;
      }
      unsigned seed;
      seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::shuffle(foo, foo + filter->metadata.range, std::default_random_engine(seed));
      printf("[CYDBG]\n");

      printf("SHUFFLED\n");

      float skewnessR = skewness / 100.0;
      uint64_t pre_val;
      std::default_random_engine generator;
      vals = (uint64_t*)malloc(nvals*sizeof(vals[0]));
      zipfian_int_distribution<uint64_t> distribution(0, nvals, skewnessR);
      for (uint64_t i = 0; i < nvals; i++) {
        pre_val = distribution(generator) % filter->metadata.range;
        vals[i] = foo[pre_val];
        uniq_vals[i] = vals[i];
      }

      printf("Zipfian Created\n");*/
    }

    /*CYDBG uniq_vals*/
//    std::sort(uniq_vals.begin(), uniq_vals.end());
//    uniq_vals.erase(std::unique(uniq_vals.begin(), uniq_vals.end()), uniq_vals.end());
    /*CYDBG*/

    other_vals = (uint64_t*)malloc(nvals*sizeof(other_vals[0]));
    mt19937 seed(50);
    uniform_int_distribution<uint64_t> dist(0, filter->metadata.range - 1);
//    RAND_bytes((unsigned char *)other_vals, sizeof(*other_vals) * nvals);
    for (uint64_t i = 0; i < nvals; i++) {
      other_vals[i] = dist(seed);
//      other_vals[i] = (1 * other_vals[i]) % filter->metadata.range;
    }

/*    ofstream outFile("vals.txt");
    if (outFile) {
      for (uint64_t i = 0; i < nvals; i++) {
        outFile << vals[i] << "\n";
      }
    }
    outFile.close();

    ofstream outFile3("other_vals.txt");
    if (outFile3) {
      for (uint64_t i = 0; i < nvals; i++) {
        outFile3 << other_vals[i] << "\n";
      }
    }
    outFile3.close();

    ofstream outFile2("uniq_vals.txt");
    if (outFile2) {
      for (uint64_t i = 0; i < uniq_vals.size(); i++) {
        outFile2 << uniq_vals[i] << " " << count(vals, vals + nvals, uniq_vals[i]) << "\n";
      }
    }
    outFile2.close();

    ifstream inFile("/home/ubuntu/filters/cvqf/vals.txt");
    if (inFile.is_open()) {
      string line;
      for (uint64_t i = 0; i < nvals; i++) {
        getline(inFile, line);
        vals[i] = stoull(line);
      }
      inFile.close();
    }

    other_vals = (uint64_t*)malloc(nvals*sizeof(other_vals[0]));
    ifstream inFile2("/home/ubuntu/filters/cvqf/other_vals.txt");
    if (inFile2.is_open()) {
      string line;
      for (uint64_t i = 0; i < nvals; i++) {
        getline(inFile2, line);
        other_vals[i] = stoull(line);
      }
      inFile2.close();
    }*/

    printf("[CYDBG] nvals: %ld\n", nvals);

    struct timeval start, end;
    struct timezone tzp;
    uint64_t elapsed_usecs;

    int insert_return = 1;

    // nanos
    //struct timespec startT, endT;

    //int put_slot = 0;
    //int all_slot = 0;

    gettimeofday(&start, &tzp);
    uint64_t num_successful_inserts = 0;
    /* Insert hashes in the vqf filter */
    for (uint64_t i = 0; i < nvals; i++) {
      insert_return = vqf_insert(filter, vals[i]);
      if (insert_return == -1) {
        fprintf(stderr, "Insertion failed\n");
        printf("[CYDBG] keys_tobe_inserted: %ld, num_succesful_inserts: %ld\n", nvals, num_successful_inserts);
        exit(EXIT_FAILURE);
      }
      num_successful_inserts++;
    }
    gettimeofday(&end, &tzp);
    printf("[CYDBG] keys_tobe_inserted: %ld, num_succesful_inserts: %ld\n", nvals, num_successful_inserts);
    elapsed_usecs = tv2usec(&end) - tv2usec(&start);
    insertion_throughput += 1.0 * nvals / elapsed_usecs;
    print_time_elapsed("Insertion time", &start, &end, nvals, "insert");
    //printf("\n%d/%d\n\n", put_slot, all_slot);

    uint64_t fslots = 0;
    for (uint64_t i = 0; i < filter->metadata.nblocks; i++) {
      uint64_t *block_md = filter->blocks[i].md;
      uint64_t lower_word = block_md[0];
      uint64_t higher_word = block_md[1];
      fslots += __builtin_popcountll(~lower_word) + __builtin_popcountll(~higher_word);
    }
    printf("[CYDBG] fslots: %ld\n", fslots);

    gettimeofday(&start, &tzp);
    /* Lookup hashes in the vqf filter (Successful Lookup) */
    for (uint64_t i = 0; i < nvals; i++) {
      if (!vqf_is_present(filter, vals[i])) {
        uint64_t block_index = vals[i] >> 8;
        uint64_t alt_block_index = ((vals[i] ^ ((vals[i] & 0xff) * 0x5bd1e995)) % filter->metadata.range) >> 8;
        fprintf(stderr, "Lookup failed for %lx, tag: %ld, i: %ld\n", vals[i], vals[i] & 0xff, i);
        exit(EXIT_FAILURE);
      }
    }
    gettimeofday(&end, &tzp);
    elapsed_usecs = tv2usec(&end) - tv2usec(&start);
    positive_throughput += 1.0 * nvals / elapsed_usecs;
      
    print_time_elapsed("Lookup time", &start, &end, nvals, "successful_lookup");
      
/*    for (uint64_t i = 0; i < uniq_vals.size(); i++) {
      printf("[CYDBG] uniq_vals[%ld]: %lx, count: %d\n", i, uniq_vals[i], get_count(filter, uniq_vals[i]));
    }*/
      
    gettimeofday(&start, &tzp);
    uint64_t nfps = 0;
    /* Lookup hashes in the vqf filter (Random Lookup) */
    for (uint64_t i = 0; i < nvals; i++) {
      if (vqf_is_present(filter, other_vals[i])) {
        nfps++;
      }
    }
    gettimeofday(&end, &tzp);
    elapsed_usecs = tv2usec(&end) - tv2usec(&start);
    negative_throughput += 1.0 * nvals / elapsed_usecs;
      
    print_time_elapsed("Random lookup:", &start, &end, nvals, "random_lookup");
    printf("%lu/%lu positives\nFP rate: 1/%f\n", nfps, nvals, 1.0 * nvals / nfps);

      
    gettimeofday(&start, &tzp);
    /* Delete hashes in the vqf filter */
    for (uint64_t i = 0; i < nvals; i++) {
      vqf_remove(filter, vals[i]);
/*      bool remove;
      remove = vqf_remove(filter, vals[i]);
      if (!remove) {
        printf("Remove failed for %ld, hash: %ld\n", i, vals[i]);
      }*/
    }
    gettimeofday(&end, &tzp);
    elapsed_usecs = tv2usec(&end) - tv2usec(&start);
    remove_throughput += 1.0 * nvals / elapsed_usecs;
      
      
    print_time_elapsed("Remove time", &start, &end, nvals, "remove");

    free(filter);
  }

  //printf("[TEST of %d]\n", TEST_NUM);
  //printf("Insertion throughput : %f Million operations / sec\n", 1.0 * insertion_throughput / TEST_NUM);
  //printf("Positive throughput  : %f Million operations / sec\n", 1.0 * positive_throughput / TEST_NUM);
  //printf("Negative throughput  : %f Million operations / sec\n", 1.0 * negative_throughput / TEST_NUM);
  //printf("Remove throughput    : %f Million operations / sec\n", 1.0 * remove_throughput / TEST_NUM);

  return 0;
}
