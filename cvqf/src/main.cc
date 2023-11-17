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
  printf("%s Total Time Elapsed: %f seconds(", desc, 1.0*elapsed_usecs / 1000000);
  if (ops) {
    printf(" %f nanoseconds/%s)", 1000.0 * elapsed_usecs / ops, opname);
    printf(" (Throughput %f Mops/sec]", 1.0 * ops / elapsed_usecs);
  }
  printf("\n");
}

void file_bit(__uint128_t num, int numbits, ofstream &File) {
  int i;
  for (i = 0; i < numbits; i++) {
    if (i != 0 && i % 8 == 0) {
      File << ":";
    }
    int a = (num >> i & 1) == 1;
    File << a;
  }
  File << "\n";
}

void file_tags(uint8_t *tags, uint32_t size, ofstream &File) {
  for (uint8_t i = 0; i < size; i++) {
    File << (uint32_t) tags[i] << " ";
  }
  File << "\n";
}

void file_block(vqf_filter * filter, uint64_t block_index, vqf_block * cur_block, ofstream &File) {
  File << "block index: " << block_index << "\n";
  File << "metadata: ";
  uint64_t *md = cur_block->md;
  file_bit(*(__uint128_t *)md, 128, File);
  File << "tags: ";
  file_tags(cur_block->tags, 48, File);
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
  printf("[CYDBG] [SIZE] vqf_block: %ld, linked_blocks: %ld, linked_list: %ld\n", sizeof(vqf_block), sizeof(linked_blocks), sizeof(linked_list));

  uint64_t qbits = atoi(argv[1]);
  uint64_t nslots = (1ULL << qbits);
//  uint64_t nvals = atoi(argv[2]);
  uint64_t load_factor = atoi(argv[2]);
  uint64_t nvals = load_factor*nslots/100;
// nvals = 40000;
  uint64_t *vals;
  uint64_t *other_vals;
  double zipf_const = std::stod(argv[3]);

  double insertion_throughput = 0.0;
  double positive_throughput = 0.0;
  double negative_throughput = 0.0;
  double remove_throughput = 0.0;
//  double count_throughput = 0.0;

//  set<uint64_t> uniq_vals;
//  vector<uint64_t> uniq_vals(nvals, 0);

  /* Repeat the test for TEST_NUM times. */
  for (int test_num = 0; test_num < TEST_NUM; test_num++) {
    vqf_filter *filter;	

    /* initialize vqf filter */
    if ((filter = vqf_init(nslots)) == NULL) {
      fprintf(stderr, "Can't allocate vqf filter.");
      exit(EXIT_FAILURE);
    }

    printf("[CYDBG] range: %ld\n", filter->metadata.range);
    

    if (zipf_const == -2) {
      nvals = 40000;
      vals = (uint64_t *)malloc(nvals * sizeof(vals[0]));
      printf("----------TESTING RESIZING----------\n");
      for (uint64_t i = 0; i < nvals; i++) {
        vals[i] = 1;
      }
    }
    else if (zipf_const == -1) {
      vals = (uint64_t *)malloc(nvals * sizeof(vals[0]));
      printf("[CYDBG] kosarak.dat used\n");
      ifstream file("/home/ubuntu/real_datasets/kosarak.dat");
//      printf("[CYDBG] vals.txt used\n");
//      ifstream file("/home/ubuntu/filters/cvqf/vals.txt");
//      printf("[CYDBG] vals.txt used\n");
//      ifstream file("/home/ubuntu/filters/cvqf/vals.txt");
      if (file.is_open()) {
        string line;
        uint64_t i = 0;
        while (i < nvals) {
          getline(file, line);
          stringstream ss(line);
          string tmp;
          while (getline(ss, tmp, ' ') && i < nvals) {
            vals[i] = stoull(tmp) * 1234567 % filter->metadata.range;
//            vals[i] = stoull(tmp) % filter->metadata.range;
//            uniq_vals[i] = vals[i];
            i++;
          }
        }
        file.close();
      }
    }
    else if (zipf_const == 0) {
      // Generate random values
      vals = (uint64_t*)malloc(nvals*sizeof(vals[0]));
      mt19937 rng(42);
      uniform_int_distribution<uint64_t> dist(0, filter->metadata.range - 1);
//      RAND_bytes((unsigned char *)vals, sizeof(*vals) * nvals);
      for (uint64_t i = 0; i < nvals; i++) {
        vals[i] = dist(rng);
//        vals[i] = (1 * vals[i]) % filter->metadata.range;
//        uniq_vals[i] = vals[i];
//        uniq_vals.insert(vals[i]);
      }
      printf("[CYDBG] Uniform Created\n");
    } else {
      Generator<uint64_t> *key_chooser_;
      key_chooser_ = new ScrambledZipfianGenerator(0, filter->metadata.range - 1, zipf_const);
      vals = (uint64_t*)malloc(nvals*sizeof(vals[0]));
      for (uint64_t i = 0; i < nvals; i++) {
        vals[i] = key_chooser_->Next() % filter->metadata.range;
//        uniq_vals[i] = vals[i];
//        uniq_vals.insert(vals[i]);
      }
      printf("Zipfian Created\n");
    }

    other_vals = (uint64_t*)malloc(nvals*sizeof(other_vals[0]));
    mt19937 seed(50);
    uniform_int_distribution<uint64_t> dist(0, filter->metadata.range - 1);
//    RAND_bytes((unsigned char *)other_vals, sizeof(*other_vals) * nvals);
    for (uint64_t i = 0; i < nvals; i++) {
//      other_vals[i] = (1 * other_vals[i]) % filter->metadata.range;
      other_vals[i] = dist(seed);
    }

    /*CYDBG uniq_vals*/
/*    std::sort(uniq_vals.begin(), uniq_vals.end());
    uniq_vals.erase(std::unique(uniq_vals.begin(), uniq_vals.end()), uniq_vals.end());*/

/*    uint64_t uniq_nvals = uniq_vals.size();
    printf("[CYDBG] nvals: %ld, uniq_nvals: %ld\n", nvals, uniq_nvals);
    uint64_t lower = pow(2, floor(log2(filter->metadata.range)));
    uint64_t upper = pow(2, ceil(log2(filter->metadata.range)));
    printf("[CYDBG] range: %ld, lower: %ld, upper: %ld\n", filter->metadata.range, lower, upper);*/
    /*CYDBG*/

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
      for (auto itr = uniq_vals.begin(); itr != uniq_vals.end(); itr++) {
        outFile2 << (*itr) << " " << count(vals, vals + nvals, (*itr)) << "\n";
      }
    }
    outFile2.close();

    ifstream inFile("vals.txt");
    if (inFile.is_open()) {
      string line;
      for (uint64_t i = 0; i < nvals; i++) {
        getline(inFile, line);
        vals[i] = stoull(line);
      }
      inFile.close();
    }

    ifstream inFile2("other_vals.txt");
    if (inFile2.is_open()) {
      string line;
      for (uint64_t i = 0; i < nvals; i++) {
        getline(inFile2, line);
        other_vals[i] = stoull(line);
      }
      inFile2.close();
    }

    vector<uint64_t> uniq_count (uniq_nvals, 0);
    for (uint64_t i = 0; i < uniq_nvals; i++) {
      uniq_count[i] = count(vals, vals + nvals, uniq_vals[i]);
    }*/


    struct timeval start, end;
    struct timezone tzp;
    uint64_t elapsed_usecs;

    uint64_t insert_return = 1;

    gettimeofday(&start, &tzp);
    uint64_t num_successful_inserts = 0;
//    ofstream insertFile("insert.txt");
    /* Insert hashes in the vqf filter */
    for (uint64_t i = 0; i < nvals; i++) {
      insert_return = vqf_insert(filter, vals[i]);
      if (insert_return == UINT64_MAX) {
        fprintf(stderr, "Insertion failed\n");
        printf("[CYDBG] keys_to_be_inserted: %ld, num_succesful_inserts: %ld\n", nvals, num_successful_inserts);
        exit(EXIT_FAILURE);
      }
      /*CYDBG*/
/*      else {
        linked_list * blocks = filter->blocks;
        linked_blocks * last_block = &blocks[insert_return].head_block;
        insertFile << "i: " << i << "\n";
        insertFile << "hash: " << vals[i] << "\n";
        insertFile << "tag: " << (vals[i] & 0xff) << "\n";
        uint64_t block_index = vals[i] >> 8;
        uint64_t tag = vals[i] & 0xff;
        uint64_t alt_block_index = ((vals[i] ^ (tag * 0x5bd1e995)) % filter->metadata.range) >> 8;
        insertFile << "block_index: " << block_index / 80 << ", alt_block_index: " << alt_block_index / 80 << "\n";
        insertFile << "block_index: " << block_index / 80;
        insertFile << "\noffset: " << block_index % 80;
        insertFile << "\nalt_block_index: " << alt_block_index / 80;
        insertFile << "\noffset: " << alt_block_index % 80<< "\n";
        do {
          file_block(filter, insert_return, &last_block->block, insertFile);
          last_block = last_block->next;
        } while (last_block != NULL);
        insertFile << "\n";
      }*/
      /*CYDBG*/
      num_successful_inserts++;
    }
//    insertFile.close();
    gettimeofday(&end, &tzp);
    printf("[CYDBG] keys_tobe_inserted: %ld, num_succesful_inserts: %ld\n", nvals, num_successful_inserts);
    elapsed_usecs = tv2usec(&end) - tv2usec(&start);
    insertion_throughput += 1.0 * nvals / elapsed_usecs;
    print_time_elapsed("Insertion time", &start, &end, nvals, "insert");
    printf("[CYDBG] SIZE: %ld, total_blocks: %ld, add_blocks: %ld\n", filter->metadata.total_size_in_bytes, filter->metadata.nblocks, filter->metadata.add_blocks);

/*    uint64_t fslots = 0;
    for (uint64_t i = 0; i < filter->metadata.nblocks; i++) {
      uint64_t *block_md = filter->blocks[i].block.md;
      uint64_t lower_word = block_md[0];
      uint64_t higher_word = block_md[1];
      fslots += __builtin_popcountll(~lower_word) + __builtin_popcountll(~higher_word);
    }
    printf("[CYDBG] fslots: %ld\n", fslots);*/

    gettimeofday(&start, &tzp);
    /* Lookup hashes in the vqf filter (Successful Lookup) */
    for (uint64_t i = 0; i < nvals; i++) {
      if (!vqf_is_present(filter, vals[i])) {
        fprintf(stderr, "Lookup failed for %lx, tag: %ld, i: %ld\n", vals[i], vals[i] & 0xff, i);
//        exit(EXIT_FAILURE);
     }
    }
    gettimeofday(&end, &tzp);
    elapsed_usecs = tv2usec(&end) - tv2usec(&start);
    positive_throughput += 1.0 * nvals / elapsed_usecs;
      
    print_time_elapsed("Lookup time", &start, &end, nvals, "successful_lookup");

//    uint64_t count_fail = 0;
//    gettimeofday(&start, &tzp);
    /*Get Count*/
//    for (uint64_t i = 0; i < uniq_nvals; i++) {
//      printf("[CYDBG] val: %ld, get_count: %d\n" , uniq_vals[i], get_count(filter, uniq_vals[i]));
//      if (get_count(filter, uniq_vals[i]) != uniq_count[i])
//        count_fail++;
/*      if (get_count(filter, vals[i]) != count(vals, vals + nvals, vals[i])) {
       if (outFile3) {
          outFile3 << "value: " << vals[i] << ", get_count: " << get_count(filter, vals[i]) << ", count: " << count(vals, vals + nvals, vals[i]) << "\n";
        }
        count_fail++;
      }
      }*/
//    gettimeofday(&end, &tzp);
//    outFile3.close();
//    elapsed_usecs = tv2usec(&end) - tv2usec(&start);
//    count_throughput += 1.0 * nvals / elapsed_usecs;
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
    printf("%lu/%lu positives\nFP rate: %f\n", nfps, nvals, 1.0 * nfps / nvals);

    printf("[CYDBG] Print Filter\n");
    ofstream filterFile("filter.txt");
    if (filterFile) {
      filterFile << "total blocks: " << filter->metadata.nblocks << "\n";
      filterFile << "add blocks: " << filter->metadata.add_blocks << "\n";
      for (uint64_t i = 0; i < filter->metadata.nblocks; i++) {
        linked_list    *blocks             = filter->blocks;
        linked_blocks *last_block = &blocks[i].head_block;
        do {
          file_block(filter, i, &last_block->block, filterFile);
          last_block = last_block->next;
        } while(last_block != NULL);
        filterFile << "\n";
      }
    }
    filterFile.close();

    gettimeofday(&start, &tzp);
    /* Delete hashes in the vqf filter */
//    ofstream removeFile("remove.txt");
//    ofstream removeFailFile("removeFail.txt");
//    linked_blocks * blocks = filter->blocks;
    for (uint64_t i = 0; i < nvals; i++) {
      vqf_remove(filter, vals[i]);
/*      bool remove;
      remove = vqf_remove(filter, vals[i]);
      if (!remove) {
        printf("Remove failed for %ld, hash: %ld\n", i, vals[i]); 
        uint64_t block_index = vals[i] >> 8;
        uint64_t tag = vals[i] & 0xff;
        uint64_t alt_block_index = ((vals[i] ^ (tag * 0x5bd1e995)) % filter->metadata.range) >> 8;
        linked_blocks *l_block = &blocks[block_index / 80];
        linked_blocks *l_block2 = &blocks[alt_block_index / 80];
        removeFailFile << "i: " << i << "\n";
        removeFailFile << "tag: " << (vals[i] & 0xff) << "\n";
        do {
          file_block(filter, block_index / 80, &l_block->block, removeFailFile);
          removeFailFile << "offset: " << block_index % 80 << "\n";
          l_block = l_block->next;
        } while (l_block != NULL);
        do {
          file_block(filter, alt_block_index / 80, &l_block2->block, removeFailFile);
          removeFailFile << "offset: " << alt_block_index % 80 << "\n";
          l_block2 = l_block2->next;
        } while (l_block2 != NULL);
        removeFailFile << "\n";
      } else {
        linked_blocks *last_block = &blocks[remove];
        removeFile << "i: " << i << "\n";
        removeFile << "tag: " <<  (vals[i] & 0xff) << "\n";
        do {
          file_block(filter, remove, &last_block->block, removeFile);
          last_block = last_block->next;
        } while (last_block != NULL);
        removeFile << "\n";
      }*/
//      printf("\n");
      //bool success;
      //success = vqf_remove(filter, vals[i]);
      //if (success == false) {
      //  fprintf(stderr, "Remove failed for %ld\n", vals[i]);
      //  exit(EXIT_FAILURE);
      //}
    }
//    removeFile.close();
//    removeFailFile.close();
    gettimeofday(&end, &tzp);
    elapsed_usecs = tv2usec(&end) - tv2usec(&start);
    remove_throughput += 1.0 * nvals / elapsed_usecs;
      
      
    print_time_elapsed("Remove time", &start, &end, nvals, "remove");


    /*Free Linked Blocks*/
    if (filter->metadata.add_blocks != 0) {
      for (uint64_t i = 0; i < filter->metadata.nblocks; i++) {
        linked_blocks *last_block = filter->blocks[i].head_block.next;
        while (last_block != NULL) {
          linked_blocks *to_be_freed = last_block;
          last_block = last_block->next;
          free(to_be_freed);
        }
      }
    }

    free(filter);
  }

  //printf("[TEST of %d]\n", TEST_NUM);
  //printf("Insertion throughput : %f Million operations / sec\n", 1.0 * insertion_throughput / TEST_NUM);
  //printf("Positive throughput  : %f Million operations / sec\n", 1.0 * positive_throughput / TEST_NUM);
  //printf("Negative throughput  : %f Million operations / sec\n", 1.0 * negative_throughput / TEST_NUM);
  //printf("Remove throughput    : %f Million operations / sec\n", 1.0 * remove_throughput / TEST_NUM);

  return 0;
}
