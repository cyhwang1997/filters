/*
 * ============================================================================
 *
 *        Authors:  Prashant Pandey <ppandey@cs.stonybrook.edu>
 *                  Rob Johnson <robj@vmware.com>   
 *
 * ============================================================================
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <openssl/rand.h>

#include "include/gqf.h"
#include "include/gqf_int.h"
#include "include/gqf_file.h"

#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <random>
#include "zipfian_int_distribution.h"
#include <time.h>
#include <sstream>

#include "utils.h"
#include "zipfian_generator.h"
#include "scrambled_zipfian_generator.h"
#include <fstream>

using namespace ycsbc;
using namespace std;

uint64_t tv2usec(struct timeval *tv){
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
	printf("\n");;
}

int main(int argc, char **argv)
{
	if (argc < 4) {
		fprintf(stderr, "Please specify three arguments: \n \
	1. Log of the number of slots \n \
	2. Load factor (0 - 95).\n \
	3. Skewness (0 - 1). \n");
		exit(1);
	}
	QF qf;
	uint64_t qbits = atoi(argv[1]);
//	uint64_t rbits = atoi(argv[2]);
	uint64_t rbits = 8;
	/*CYDBG*/
	uint64_t load_factor = atoi(argv[2]);
	double zipf_const = std::stod(argv[3]);
	/*CYDBG*/
	uint64_t nhashbits = qbits + rbits;
	uint64_t nslots = (1ULL << qbits);
	uint64_t nvals = load_factor*nslots/100;
	uint64_t key_count = 1; /* CYDBG: originally 4 */
	uint64_t *vals;
//	vector<uint64_t> vals(nvals, 0);
	uint64_t *other_vals;
//  std::set<uint64_t> uniq_vals;

	printf("[CYDBG] nvals: %ld, nslots: %ld\n", nvals, nslots);

	/* Initialise the CQF */
	/*if (!qf_malloc(&qf, nslots, nhashbits, 0, QF_HASH_INVERTIBLE, 0)) {*/
	/*fprintf(stderr, "Can't allocate CQF.\n");*/
	/*abort();*/
	/*}*/
	if (!qf_initfile(&qf, nslots, nhashbits, 0, QF_HASH_INVERTIBLE, 0,
									 "mycqf.file")) {
		fprintf(stderr, "Can't allocate CQF.\n");
		abort();
	}

	qf_set_auto_resize(&qf, true); /*true to false*/

	if (zipf_const == -1) {
    vals = (uint64_t *)malloc(nvals * sizeof(vals[0]));
//    printf("[CYDBG] caida used\n");
//    ifstream file("/home/ubuntu/real_datasets/caida/caida_ip.txt");
    printf("[CYDBG] webdocs.dat used\n");
    ifstream file("/home/ubuntu/real_datasets/fimi/webdocs.dat");
    if (file.is_open()) {
      string line;
      uint64_t i = 0;
      while (i < nvals) {
        getline(file, line);
        stringstream ss(line);
        string tmp;
        while (getline(ss, tmp, ' ') && i < nvals) {
          vals[i] = stoull(tmp) * 1234567 % qf.metadata->range;
//          vals[i] = stoull(tmp);
          i++;
        }
      }
      file.close();
    }
  } else if(zipf_const == 0) {
		/* Generate random values */
		vals = (uint64_t*)malloc(nvals*sizeof(vals[0]));
    mt19937 rng(42);
    uniform_int_distribution<uint64_t> dist(0, qf.metadata->range - 1);
    for (uint64_t i = 0; i < nvals; i++) {
      vals[i] = dist(rng);
    }
    printf("[CYDBG] Uniform Created\n");
/*		RAND_bytes((unsigned char *)vals, sizeof(*vals) * nvals);
		srand(0);
		for (uint64_t i = 0; i < nvals; i++) {
			vals[i] = (1 * vals[i]) % qf.metadata->range;
			vals[i] = rand() % qf.metadata->range;
			fprintf(stdout, "%lx\n", vals[i]);
		}*/
	} else {
    printf("Creating Zipfian\n");
    Generator<uint64_t> *key_chooser_;
    key_chooser_ = new ScrambledZipfianGenerator(0, qf.metadata->range - 1, zipf_const);
    vals = (uint64_t *)malloc(nvals * sizeof(vals[0]));
    for (uint64_t i = 0; i < nvals; i++) {
      vals[i] = key_chooser_->Next() % qf.metadata->range;
    }
    printf("Zipfian Created\n");
	}
  printf("[CYDBG] range: %ld\n", qf.metadata->range);

	other_vals = (uint64_t*)malloc(nvals*sizeof(other_vals[0]));
  mt19937 seed(50);
  uniform_int_distribution<uint64_t> dist(0, qf.metadata->range - 1);

  for (uint64_t i = 0; i < nvals; i++) {
/*    if (find(vals, vals+nvals, dist(seed)) != vals+nvals) {
      i--;
      continue;
    }*/
    other_vals[i] = dist(seed);
  }

	double insertion_throughput = 0.0;
	double positive_throughput = 0.0;
	double negative_throughput = 0.0;
	double remove_throughput = 0.0;

	struct timeval start, end;
	struct timezone tzp;
	uint64_t elapsed_usecs;

	printf("[CYDBG] Inserting keys\n");
	gettimeofday(&start, &tzp);
  /*CYDBG*/
	/* Insert keys in the CQF */
	for (uint64_t i = 0; i < nvals; i++) {
		int ret = qf_insert(&qf, vals[i] % qf.metadata->range, 0, key_count, QF_NO_LOCK);
		if (ret < 0) {
			fprintf(stderr, "failed insertion for key: %lx %d.\n", vals[i], 50);
			if (ret == QF_NO_SPACE)
				fprintf(stderr, "CQF is full.\n");
			else if (ret == QF_COULDNT_LOCK)
				fprintf(stderr, "TRY_ONCE_LOCK failed.\n");
			else
				fprintf(stderr, "Does not recognise return value.\n");
			abort();
		}
	}
  /*CYDBG*/
	gettimeofday(&end, &tzp);
  elapsed_usecs = tv2usec(&end) - tv2usec(&start);
	insertion_throughput += 1.0 * nvals / elapsed_usecs;
	print_time_elapsed("Insertion time:", &start, &end, nvals, "insert");

	printf("[CYDBG] total_size_in_bytes: %ld\n", qf_get_total_size_in_bytes(&qf));

	printf("[CYDBG] Looking up keys\n");
	gettimeofday(&start, &tzp);
//  uint64_t count_fail = 0;
  /*CYDBG*/
	/* Lookup inserted keys and counts. */
	for (uint64_t i = 0; i < nvals; i++) {
		int count = qf_count_key_value(&qf, vals[i] % qf.metadata->range, 0, 0);
/*    if (count != std::count(vals.begin(), vals.end(), vals[i])) {
      count_fail++;
    }*/
/*		if (count < key_count) {
			fprintf(stderr, "failed lookup after insertion for %lx %ld.\n", vals[i],
							count);
			abort();
		}*/
	}
  /*CYDBG*/
	gettimeofday(&end, &tzp);
  elapsed_usecs = tv2usec(&end) - tv2usec(&start);
	positive_throughput += 1.0 * nvals / elapsed_usecs;
	print_time_elapsed("Lookup time:", &start, &end, nvals, "insert");
//  printf("[CYDBG] count_fail: %ld\n", count_fail);

	gettimeofday(&start, &tzp);
	uint64_t nfps = 0;
	/* Lookup hashes (Random Lookup) */
	for (uint64_t i = 0; i < nvals; i++) {
		uint64_t count = qf_count_key_value(&qf, other_vals[i] % qf.metadata->range, 0, 0);
		if (count != 0)
			nfps++;
	}
	gettimeofday(&end, &tzp);
	elapsed_usecs = tv2usec(&end) - tv2usec(&start);
	negative_throughput += 1.0 * nvals / elapsed_usecs;

	print_time_elapsed("Random lookup:", &start, &end, nvals, "insert");
	printf("%lu/%lu positives\nFP rate: %f\n", nfps, nvals, 1.0 * nfps / nvals);
	/*CYDBG*/

#if 0
	for (uint64_t i = 0; i < nvals; i++) {
		uint64_t count = qf_count_key_value(&qf, vals[i], 0, 0);
		if (count < key_count) {
			fprintf(stderr, "failed lookup during deletion for %lx %ld.\n", vals[i],
							count);
			abort();
		}
		if (count > 0) {
			/*fprintf(stdout, "deleting: %lx\n", vals[i]);*/
			qf_delete_key_value(&qf, vals[i], 0, QF_NO_LOCK);
			/*qf_dump(&qf);*/
			uint64_t cnt = qf_count_key_value(&qf, vals[i], 0, 0);
			if (cnt > 0) {
				fprintf(stderr, "failed lookup after deletion for %lx %ld.\n", vals[i],
								cnt);
				abort();
			}
		}
	}
#endif

	/* Write the CQF to disk and read it back. */
/*	char filename[] = "mycqf_serialized.cqf";
	fprintf(stdout, "Serializing the CQF to disk.\n");
	uint64_t total_size = qf_serialize(&qf, filename);
	if (total_size < sizeof(qfmetadata) + qf.metadata->total_size_in_bytes) {
		fprintf(stderr, "CQF serialization failed.\n");
		abort();
	}
	qf_deletefile(&qf);

	QF file_qf;
	fprintf(stdout, "Reading the CQF from disk.\n");
	if (!qf_deserialize(&file_qf, filename)) {
		fprintf(stderr, "Can't initialize the CQF from file: %s.\n", filename);
		abort();
	}
	for (uint64_t i = 0; i < nvals; i++) {
		uint64_t count = qf_count_key_value(&file_qf, vals[i], 0, 0);
		if (count < key_count) {
			fprintf(stderr, "failed lookup in file based CQF for %lx %ld.\n",
							vals[i], count);
			abort();
		}
	}

	fprintf(stdout, "Testing iterator and unique indexes.\n");*/
	/* Initialize an iterator and validate counts. */
/*	QFi qfi;
	qf_iterator_from_position(&file_qf, &qfi, 0);
	QF unique_idx;
	if (!qf_malloc(&unique_idx, file_qf.metadata->nslots, nhashbits, 0,
								 QF_HASH_INVERTIBLE, 0)) {
		fprintf(stderr, "Can't allocate set.\n");
		abort();
	}

	int64_t last_index = -1;
	int i = 0;
	qf_iterator_from_position(&file_qf, &qfi, 0);
	while(!qfi_end(&qfi)) {
		uint64_t key, value, count;
		qfi_get_key(&qfi, &key, &value, &count);
		if (count < key_count) {
			fprintf(stderr, "Failed lookup during iteration for: %lx. Returned count: %ld\n",
							key, count);
			abort();
		}
		int64_t idx = qf_get_unique_index(&file_qf, key, value, 0);
		if (idx == QF_DOESNT_EXIST) {
			fprintf(stderr, "Failed lookup for unique index for: %lx. index: %ld\n",
							key, idx);
			abort();
		}
		if (idx <= last_index) {
			fprintf(stderr, "Unique indexes not strictly increasing.\n");
			abort();
		}
		last_index = idx;
		if (qf_count_key_value(&unique_idx, key, 0, 0) > 0) {
			fprintf(stderr, "Failed unique index for: %lx. index: %ld\n",
							key, idx);
			abort();
		}
		qf_insert(&unique_idx, key, 0, 1, QF_NO_LOCK);
		int64_t newindex = qf_get_unique_index(&unique_idx, key, 0, 0);
		if (idx < newindex) {
			fprintf(stderr, "Index weirdness: index %dth key %ld was at %ld, is now at %ld\n",
							i, key, idx, newindex);
			//abort();
		}

		i++;
		qfi_next(&qfi);
	}*/

	/*CYDBG*/
	gettimeofday(&start, &tzp);
	/*CYDBG*/
	/* remove some counts  (or keys) and validate. */
	fprintf(stdout, "Testing remove/delete_key.\n");
	for (uint64_t i = 0; i < nvals; i++) {
		/*uint64_t count = qf_count_key_value(&qf, vals[i] % qf.metadata->range, 0, 0); //CY: file_qf->qf
		if (count < key_count) {
		fprintf(stderr, "failed lookup during deletion for %lx %ld.\n", vals[i],
		count);
		abort();
		}*/
    qf_remove(&qf, vals[i], 0, 1, QF_NO_LOCK);
/*		int ret = qf_delete_key_value(&qf, vals[i] % qf.metadata->range, 0, QF_NO_LOCK); //CY: file_qf->qf
		uint64_t count = qf_count_key_value(&qf, vals[i] % qf.metadata->range, 0, 0); //CY: file_qf->qf
		if (count > 0) {
			if (ret < 0) {
				fprintf(stderr, "failed deletion for %lx %ld ret code: %d.\n",
								vals[i], count, ret);
				abort();
			}
			uint64_t new_count = qf_count_key_value(&qf, vals[i] % qf.metadata->range, 0, 0); //CY: file_qf->qf
			if (new_count > 0) {
				fprintf(stderr, "delete key failed for %lx %ld new count: %ld.\n",
								vals[i], count, new_count);
				abort();
			}
		}*/
	}
	/*CYDBG*/
	gettimeofday(&end, &tzp);
	elapsed_usecs = tv2usec(&end) - tv2usec(&start);
	remove_throughput += 1.0 * nvals / elapsed_usecs;

	print_time_elapsed("Remove time", &start, &end, nvals, "remove");
	/*CYDBG*/

//	qf_deletefile(&file_qf);

	fprintf(stdout, "Validated the CQF.\n");
}

