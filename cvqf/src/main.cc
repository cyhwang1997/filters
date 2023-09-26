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
#include <sstream>

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
//  uint64_t nvals = atoi(argv[2]);
  uint64_t load_factor = atoi(argv[2]);
  uint64_t nvals = load_factor*nslots/100;
/*CY*/ nvals = 40000;
  uint64_t *vals;
  uint64_t *other_vals;
  double zipf_const = std::stod(argv[3]);

//  uint64_t sizeval = atoi(argv[4]);
//  double sizevalR = 1.0 * sizeval / 100;

  double insertion_throughput = 0.0;
  double positive_throughput = 0.0;
  double negative_throughput = 0.0;
  double remove_throughput = 0.0;
  double count_throughput = 0.0;

//  set<uint64_t> uniq_vals;
  vector<uint64_t> uniq_vals(nvals, 0);

/*
#if TAG_BITS == 8
   uint64_t filterMetadataRange = (nslots + 48)/48;
   filterMetadataRange = filterMetadataRange * 80 * (1ULL << TAG_BITS);
#elif TAG_BITS == 16
   uint64_t filterMetadataRange = (nslots + 28)/28;
   filterMetadataRange = filterMetadataRange * 36 * (1ULL << TAG_BITS);
#endif
   printf("filter->metadata.range = %ld, nvals = %ld, nslots = %ld\n", filterMetadataRange, nvals, nslots);
*/

  /* Repeat the test for TEST_NUM times. */
  for (int test_num = 0; test_num < TEST_NUM; test_num++) {
    vqf_filter *filter;	

    /* initialize vqf filter */
    if ((filter = vqf_init(nslots)) == NULL) {
      fprintf(stderr, "Can't allocate vqf filter.");
      exit(EXIT_FAILURE);
    }
    

    if (zipf_const == -2) {
      nvals = 40000;
      vals = (uint64_t *)malloc(40000 * sizeof(vals[0]));
      printf("----------TESTING RESIZING----------\n");
/*      for (uint64_t i = 0; i < 20000; i++)
        vals[i] = 1;*/
      for (uint64_t i = 0; i < 30000; i++)
        vals[i] = 1;
      for (uint64_t i = 30000; i < 40000; i++)
        vals[i] = 2;
      uniq_vals.resize(2);
      uniq_vals[0] = 1;
      uniq_vals[1] = 2;
    }
    else if (zipf_const == -1) {
      vals = (uint64_t *)malloc(nvals * sizeof(vals[0]));
      printf("[CYDBG] kosarak.dat used\n");
      ifstream file("/home/ubuntu/real_datasets/kosarak.dat");
      if (file.is_open()) {
        string line;
        uint64_t i = 0;
        while (i < nvals) {
          getline(file, line);
          stringstream ss(line);
          string tmp;
          while (getline(ss, tmp, ' ') && i < nvals) {
            vals[i] = stoull(tmp);
            uniq_vals[i] = vals[i];
            i++;
          }
        }
        file.close();
      }
    }
    else if (zipf_const == 0) {
      // Generate random values
      vals = (uint64_t*)malloc(nvals*sizeof(vals[0]));
      RAND_bytes((unsigned char *)vals, sizeof(*vals) * nvals);
      for (uint64_t i = 0; i < nvals; i++) {
        vals[i] = (1 * vals[i]) % filter->metadata.range;
        uniq_vals[i] = vals[i];
//        uniq_vals.insert(vals[i]);
      }
      printf("[CYDBG] Uniform Created\n");
    } else {
      Generator<uint64_t> *key_chooser_;
      key_chooser_ = new ScrambledZipfianGenerator(0, filter->metadata.range, zipf_const);
      vals = (uint64_t*)malloc(nvals*sizeof(vals[0]));
      for (uint64_t i = 0; i < nvals; i++) {
        vals[i] = key_chooser_->Next() % filter->metadata.range;
        uniq_vals[i] = vals[i];
//        uniq_vals.insert(vals[i]);
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
    std::sort(uniq_vals.begin(), uniq_vals.end());
    uniq_vals.erase(std::unique(uniq_vals.begin(), uniq_vals.end()), uniq_vals.end());
    /*CYDBG*/

/*    ofstream outFile("vals.txt");
    if (outFile) {
      for (uint64_t i = 0; i < nvals; i++) {
        outFile << vals[i] << "\n";
      }
    }
    outFile.close();

    ofstream outFile2("uniq_vals.txt");
    if (outFile2) {
      for (auto itr = uniq_vals.begin(); itr != uniq_vals.end(); itr++) {
        outFile2 << (*itr) << " " << count(vals, vals + nvals, (*itr)) << "\n";
      }
    }
    outFile2.close();*/

/*    ifstream file("input.txt");
    if (file.is_open()) {
      string line;
      for (uint64_t i = 0; i < nvals; i++) {
        getline(file, line);
        vals[i] = stoull(line);
      }
      file.close();
    }*/

    uint64_t uniq_nvals = uniq_vals.size();
    printf("[CYDBG] nvals: %ld, uniq_nvals: %ld\n", nvals, uniq_nvals);

    vector<uint64_t> uniq_count (uniq_nvals, 0);
    for (uint64_t i = 0; i < uniq_nvals; i++) {
      uniq_count[i] = count(vals, vals + nvals, uniq_vals[i]);
    }


    other_vals = (uint64_t*)malloc(nvals*sizeof(other_vals[0]));
    RAND_bytes((unsigned char *)other_vals, sizeof(*other_vals) * nvals);
    for (uint64_t i = 0; i < nvals; i++) {
      other_vals[i] = (1 * other_vals[i]) % filter->metadata.range;
    }

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
        printf("[CYDBG] val: %d, get_count: %d\n" , 111009, get_count(filter, 111009));
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
      uint64_t *block_md = filter->blocks[i].block.md;
      uint64_t lower_word = block_md[0];
      uint64_t higher_word = block_md[1];
      fslots += __builtin_popcountll(~lower_word) + __builtin_popcountll(~higher_word);
    }
    printf("[CYDBG] fslots: %ld\n", fslots);


    gettimeofday(&start, &tzp);
    /* Lookup hashes in the vqf filter (Successful Lookup) */
    printf("LOOKUP 1\n");
    if (vqf_is_present(filter, 1))
      printf("true\n");
    else
      printf("false\n");
    printf("LOOKUP 2\n");
    if (vqf_is_present(filter, 2))
      printf("true\n");
    else
      printf("false\n");

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

//    uint64_t count_fail = 0;
    gettimeofday(&start, &tzp);
    /*Get Count*/
//    for (auto i :uniq_vals) {
    for (uint64_t i = 0; i < uniq_nvals; i++) {
      printf("[CYDBG] val: %ld, get_count: %d\n" , uniq_vals[i], get_count(filter, uniq_vals[i]));
//      if (get_count(filter, uniq_vals[i]) != uniq_count[i])
//        count_fail++;
/*      if (get_count(filter, vals[i]) != count(vals, vals + nvals, vals[i])) {
       if (outFile3) {
          outFile3 << "value: " << vals[i] << ", get_count: " << get_count(filter, vals[i]) << ", count: " << count(vals, vals + nvals, vals[i]) << "\n";
        }
        count_fail++;
      }*/
    }
    gettimeofday(&end, &tzp);
//    outFile3.close();
    elapsed_usecs = tv2usec(&end) - tv2usec(&start);
    count_throughput += 1.0 * nvals / elapsed_usecs;
//    printf("[CYDBG] count_fail: %ld\n", count_fail);
      
//    print_time_elapsed("Get Count time", &start, &end, uniq_nvals, "count_time");
      
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
      bool remove = vqf_remove(filter, vals[i]);
      uint64_t block_index = vals[i] >> 8;
      uint64_t alt_block_index = ((vals[i] ^ ((vals[i] & 0xff) * 0x5bd1e995)) % filter->metadata.range) >> 8;
      printf("hash: %ld, i: %ld\n", vals[i], i);
      print_block(filter, block_index / 80, &filter->blocks[block_index / 80].block);
      print_block(filter, alt_block_index / 80, &filter->blocks[alt_block_index / 80].block);
      printf("\n");
      if (!remove) {
        printf("Remove failed for %ld, hash: %ld\n", i, vals[i, vals[i]]);
        print_block(filter, 0, &filter->blocks[0].block);
      }
      //bool success;
      //success = vqf_remove(filter, vals[i]);
      //if (success == false) {
      //  fprintf(stderr, "Remove failed for %ld\n", vals[i]);
      //  exit(EXIT_FAILURE);
      //}
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
