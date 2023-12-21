#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <openssl/rand.h>
#include <set>
#include <sys/time.h>
#include <math.h>
#include <time.h>
#include <vector>
#include <cstring>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <random>
#include <memory>
#include <fstream>
#include <sstream>

#include "bf/all.hpp"
#include "utils.h"
#include "zipfian_generator.h"
#include "scrambled_zipfian_generator.h"

using namespace bf;
using namespace std;
using namespace ycsbc;

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
  if (argc < 5) {
    fprintf(stderr, "Specify three arguments: \n \
  1. number of cells \n \
  2. bits per cell \n \
  3. number of hash functions \n \
  4. load factor\n \
  5. Zipfian constant(double)\n \
  6. range \n");

    exit(1);
  }

  uint64_t num_cells = atoi(argv[1]);
  uint64_t bits_per_cell = atoi(argv[2]);
  uint64_t num_hashes = atoi(argv[3]);
  uint64_t load_factor = atoi(argv[4]);
  uint64_t nvals = load_factor * num_cells/100;
  uint64_t *vals;
  uint64_t *other_vals;
  double zipf_const = stod(argv[5]);
  uint64_t range = atoi(argv[6]);

  double insertion_throughput = 0.0;
  double positive_throughput = 0.0;
  double negative_throughput = 0.0;
  double remove_throughput = 0.0;
  
//  std::unique_ptr<bloom_filter> bf;
//  auto h = make_hasher(num_hashes, 0, false);
//  bf.reset(new counting_bloom_filter(std::move(h), num_cells, bits_per_cell, false));
  counting_bloom_filter bf(make_hasher(num_hashes, 0, false), num_cells, bits_per_cell, false);

  if (zipf_const == -1) {
    vals = (uint64_t*)malloc(nvals * sizeof(vals[0]));
//    printf("[CYDBG] caida used\n");
//    ifstream file("/home/ubuntu/real_datasets/caida/caida_ip.txt");
    printf("[CYDBG] webdocs used\n");
    ifstream file("/home/ubuntu/real_datasets/webdocs.dat");
    if (file.is_open()) {
      string line;
      uint64_t i = 0;
      while (i < nvals) {
        getline(file, line);
        stringstream ss(line);
        string tmp;
        while (getline(ss, tmp, ' ') && i < nvals) {
          vals[i] = stoull(tmp) % range;
          i++;
        }
      }
      file.close();
    }
  } else if (zipf_const == 0) {
    vals = (uint64_t*)malloc(nvals * sizeof(vals[0]));
    mt19937 rng(42);
    uniform_int_distribution<uint64_t> dist(0, range - 1);
    for (uint64_t i = 0; i < nvals; i++) {
      vals[i] = dist(rng);
    }
    printf("[CYDBG] Uniform Created\n");
  } else {
    printf("[CYDBG] Creating Zipfian\n");
    Generator<uint64_t> *key_chooser_;
    key_chooser_ = new ScrambledZipfianGenerator(0, range - 1, zipf_const);
    vals = (uint64_t*)malloc(nvals * sizeof(vals[0]));
    for (uint64_t i = 0; i < nvals; i++) {
      vals[i] = key_chooser_->Next() % range;
    }
    printf("Zipfian Created\n");
  }

  other_vals = (uint64_t*)malloc(nvals * sizeof(other_vals[0]));
  mt19937 seed(50);
  uniform_int_distribution<uint64_t> dist(0, range - 1);
  for (uint64_t i = 0; i < nvals; i++) {
    other_vals[i] = dist(seed);
  }
//  RAND_bytes((unsigned char*)other_vals, sizeof(*other_vals) * nvals);


  printf("[CYDBG] INFO\n");
  printf("[CYDBG] num_cells: %ld, bits_per_cell: %ld, num_hashes: %ld, load_factor: %ld, nvals: %ld\n", num_cells, bits_per_cell, num_hashes, load_factor, nvals);
  printf("[CYDBG] bf size: %ld Bytes\n", bf.size());


  struct timeval start, end;
  struct timezone tzp;
  uint64_t elapsed_usecs;

  gettimeofday(&start, &tzp);
  for (uint64_t i = 0; i < nvals; i++) {
//    std::cout << vals[i] << std::endl;
    bf.add(vals[i]);
  }
  gettimeofday(&end, &tzp);
  elapsed_usecs = tv2usec(&end) - tv2usec(&start);
  insertion_throughput += 1.0 * nvals / elapsed_usecs;
  print_time_elapsed("Insertion time", &start, &end, nvals, "insert");

  gettimeofday(&start, &tzp);
  /* Lookup hashes in the vqf filter (Successful Lookup) */
  for (uint64_t i = 0; i < nvals; i++) {
//    std::cout << vals[i] << std::endl;
    size_t count = bf.lookup(vals[i]);
    if (count == 0) {
      std::cout << "Lookup failed for " << vals[i] << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  gettimeofday(&end, &tzp);
  elapsed_usecs = tv2usec(&end) - tv2usec(&start);
  positive_throughput += 1.0 * nvals / elapsed_usecs;
  print_time_elapsed("Lookup time", &start, &end, nvals, "successful_lookup");

  gettimeofday(&start, &tzp);
  uint64_t fp = 0;
  /* Lookup hashes in the vqf filter (Random Lookup) */
  for (uint64_t i = 0; i < nvals; i++) {
    if (bf.lookup(other_vals[i]) != 0)
      fp++;
  }
  gettimeofday(&end, &tzp);
  elapsed_usecs = tv2usec(&end) - tv2usec(&start);
  negative_throughput += 1.0 * nvals / elapsed_usecs;
  print_time_elapsed("Random lookup:", &start, &end, nvals, "random_lookup");
  printf("%ld/%ld positives\nFP rate: %f\n", fp, nvals, 1.0*fp/nvals);
  
/*  for (uint64_t i = 0; i < 10; i++) {
    bf->add(std::to_string(i));
  }*/

  gettimeofday(&start, &tzp);
  /*Remove*/
  for (uint64_t i = 0; i < nvals; i++) {
    bf.remove(vals[i]);
  }
  gettimeofday(&end, &tzp);
  elapsed_usecs = tv2usec(&end) - tv2usec(&start);
  remove_throughput += 1.0 * nvals / elapsed_usecs;
  print_time_elapsed("Remove time", &start, &end, nvals, "remove");
}
