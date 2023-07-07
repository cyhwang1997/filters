#include "cuckoofilter.h"

#include <assert.h>
#include <math.h>

#include <iostream>
#include <vector>

#include <sys/time.h>
#include <time.h>
#include <chrono>
#include <openssl/rand.h>

//FOR ZIPFIAN
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <random>
#include "zipfian_int_distribution.h"

uint64_t tv2usec(struct timeval *tv) {
   return 1000000 * tv->tv_sec + tv->tv_usec;
}

using cuckoofilter::CuckooFilter;

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "1. nslots\n2. load factor.\n3.skeweness (0 - 99)\n");
    exit(1);
  }
  //size_t total_items = 1000000;
//  size_t total_items = 2097152; /*[CYDBG] 2^21 slots*/
 // size_t total_items = 131072;  /*[CYDBG] 2^17 slots*/
  uint64_t nslots = atoi(argv[1]);
  size_t total_items = 1ULL << nslots;

  uint64_t load_factor = atoi(argv[2]);
  uint64_t skewness = atoi(argv[3]);
	uint64_t nvals = load_factor*total_items/100;
	uint64_t *vals;

  // Create a cuckoo filter where each item is of type size_t and
  // use 12 bits for each item:
  //    CuckooFilter<size_t, 12> filter(total_items);
  // To enable semi-sorting, define the storage of cuckoo filter to be
  // PackedTable, accepting keys of size_t type and making 13 bits
  // for each key:
  //   CuckooFilter<size_t, 13, cuckoofilter::PackedTable> filter(total_items);
  //CuckooFilter<size_t, 12> filter(total_items);
  CuckooFilter<uint64_t, 8> filter(total_items);
  printf("[CYDBG] filter size: %lu\n", filter.SizeInBytes());

  struct timeval start, end;
  struct timezone tzp;
  uint64_t elapsed_usecs;
  double insertion_throughput;
  double positive_throughput;
  double negative_throughput;
  double remove_throughput;

  if (skewness == 0) {
    vals = (uint64_t*)malloc(total_items*sizeof(vals[0]));
    RAND_bytes((unsigned char *)vals, sizeof(*vals)*total_items);
  } else if (skewness > 99) {
    fprintf(stderr, "Such skewness not allowed\n");
    exit(EXIT_FAILURE);
  } else {
    uint64_t *foo;
    foo = (uint64_t*)malloc((total_items)*sizeof(foo[0]));
    for (uint64_t i = 0; i < total_items; i++) {
      foo[i] = i;
    }
    unsigned seed;
    seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(foo, foo + total_items, std::default_random_engine(seed));
    printf("SHUFFLED\n");
    float skewnessR = skewness / 100.0;
    uint64_t pre_val;
    std::default_random_engine generator;
    vals = (uint64_t*)malloc(nvals * sizeof(vals[0]));
    zipfian_int_distribution<uint64_t> distribution(0, nvals, skewnessR);
    for (uint64_t i = 0; i < nvals; i++) {
      pre_val = distribution(generator) % total_items;
      vals[i] = foo[pre_val];
    }

    printf("Zipfian Created\n");
  }

/*  for (uint64_t i = 0; i < nvals; i++) {
    printf("%ld\n", vals[i]);
  }*/

  gettimeofday(&start, &tzp);
  // Insert items to this cuckoo filter
  size_t num_inserted = 0;
  for (size_t i = 0; i < load_factor*total_items / 100; i++, num_inserted++) {
    if (filter.Add(vals[i]) != cuckoofilter::Ok) {
      break;
    }
  }
  gettimeofday(&end, &tzp);
  elapsed_usecs = tv2usec(&end) - tv2usec(&start);
  insertion_throughput = 1.0 * num_inserted / elapsed_usecs;
  
  printf("Insertion Throughput: %f Million operations / sec\n", insertion_throughput);


  gettimeofday(&start, &tzp);
  // Check if previously inserted items are in the filter, expected
  // true for all items
  for (size_t i = 0; i < num_inserted; i++) {
    assert(filter.Contain(vals[i]) == cuckoofilter::Ok);
  }
  gettimeofday(&end, &tzp);
  elapsed_usecs = tv2usec(&end) - tv2usec(&start);
  positive_throughput = 1.0 * num_inserted / elapsed_usecs;

  printf("Positive Throughput: %f Million operations / sec\n", positive_throughput);

  // Check non-existing items, a few false positives expected
  gettimeofday(&start, &tzp);
  size_t total_queries = 0;
  size_t false_queries = 0;
  for (size_t i = 0; i < num_inserted; i++) {
  // for (size_t i = total_items; i < 2 * total_items; i++) {
    if (filter.Contain(i) == cuckoofilter::Ok) {
      false_queries++;
    }
    total_queries++;
  }
  gettimeofday(&end, &tzp);
  elapsed_usecs = tv2usec(&end) - tv2usec(&start);
  negative_throughput = 1.0 * num_inserted / elapsed_usecs;

  printf("Negative Throughput: %f Million operations / sec\n", negative_throughput);

  // Output the measured false positive rate
  std::cout << "false positive rate is "
            << 100.0 * false_queries / total_queries << "%\n";

  gettimeofday(&start, &tzp);
  for (size_t i = 0; i < num_inserted; i++) {
     assert(filter.Delete(vals[i]) == cuckoofilter::Ok);
  }
  gettimeofday(&end, &tzp);
  elapsed_usecs = tv2usec(&end) - tv2usec(&start);
  remove_throughput = 1.0 * num_inserted / elapsed_usecs;

  printf("Remove Throughput: %f Million operations / sec\n", remove_throughput);

  return 0;
}
