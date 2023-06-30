/*
 * test.cc
 *
 * Copyright (c) 2014 Vedant Kumar <vsk@berkeley.edu>
 */

extern "C" {
  #include "qf.c"
}

#define QBENCH 1

#include <set>
#include <vector>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <sys/time.h>
#include <openssl/rand.h>

// FOR ZIPFIAN
#include <algorithm>
#include <array>
#include <chrono>
#include <random>
#include "zipfian_int_distribution.h"

using namespace std;

uint64_t tv2usec(struct timeval *tv) {
   return 1000000 * tv->tv_sec + tv->tv_usec;
}

/* I need a more powerful machine to increase these parameters... */
const uint32_t Q_MAX = 12;
const uint32_t R_MAX = 6;
const uint32_t ROUNDS_MAX = 1000;

static void fail(struct quotient_filter *qf, const char *s)
{
  fprintf(stderr, "qf(q=%u, r=%u): %s\n", qf->qf_qbits, qf->qf_rbits, s);
  abort();
}

static uint64_t rand64()
{
  return (((uint64_t) rand()) << 32) | ((uint64_t) rand());
}

static void qf_print(struct quotient_filter *qf)
{
  char buf[32];
  uint32_t pad = uint32_t(ceil(float(qf->qf_qbits) / logf(10.f))) + 1;

  for (uint32_t i = 0; i < pad; ++i) {
    printf(" ");
  }
  printf("| is_shifted | is_continuation | is_occupied | remainder"
      " nel=%u\n", qf->qf_entries);

  for (uint64_t idx = 0; idx < qf->qf_max_size; ++idx) {
    snprintf(buf, sizeof(buf), "%lu", idx);
    printf("%s", buf);

    int fillspace = pad - strlen(buf);
    for (int i = 0; i < fillspace; ++i) {
      printf(" ");
    }
    printf("| ");

    uint64_t elt = get_elem(qf, idx);
    printf("%d          | ", !!is_shifted(elt));
    printf("%d               | ", !!is_continuation(elt));
    printf("%d           | ", !!is_occupied(elt));
    printf("%lu\n", get_remainder(elt));
  }
}

/* Check QF structural invariants. */
static void qf_consistent(struct quotient_filter *qf)
{
  assert(qf->qf_qbits);
  assert(qf->qf_rbits);
  assert(qf->qf_qbits + qf->qf_rbits <= 64);
  assert(qf->qf_elem_bits == (qf->qf_rbits + 3));
  assert(qf->qf_table);

  uint64_t idx;
  uint64_t start;
  uint64_t size = qf->qf_max_size;
  assert(qf->qf_entries <= size);
  uint64_t last_run_elt;
  uint64_t visited = 0;

  if (qf->qf_entries == 0) {
    for (start = 0; start < size; ++start) {
      assert(get_elem(qf, start) == 0);
    }
    return;
  }

  for (start = 0; start < size; ++start) {
    if (is_cluster_start(get_elem(qf, start))) {
      break;
    }
  }

  assert(start < size);

  idx = start;
  do {
    uint64_t elt = get_elem(qf, idx);

    /* Make sure there are no dirty entries. */
    if (is_empty_element(elt)) {
      assert(get_remainder(elt) == 0);
    }

    /* Check for invalid metadata bits. */
    if (is_continuation(elt)) {
      assert(is_shifted(elt));

      /* Check that this is actually a continuation. */
      uint64_t prev = get_elem(qf, decr(qf, idx));
      assert(!is_empty_element(prev));
    }

    /* Check that remainders within runs are sorted. */
    if (!is_empty_element(elt)) {
      uint64_t rem = get_remainder(elt);
      if (is_continuation(elt)) {
        assert(rem > last_run_elt);
      }
      last_run_elt = rem;
      ++visited;
    }

    idx = incr(qf, idx);
  } while (idx != start);

  assert(qf->qf_entries == visited);
}

/* Generate a random 64-bit hash. If @clrhigh, clear the high (64-p) bits. */
static uint64_t genhash(struct quotient_filter *qf, bool clrhigh,
    set<uint64_t> &keys)
{
  uint64_t hash;
  uint64_t mask = clrhigh ? LOW_MASK(qf->qf_qbits + qf->qf_rbits) : ~0ULL;
  uint64_t size = qf->qf_max_size;

  /* If the QF is overloaded, use a linear scan to find an unused hash. */
  if (keys.size() > (3 * (size / 4))) {
    uint64_t probe;
    uint64_t start = rand64() & qf->qf_index_mask;
    for (probe = incr(qf, start); probe != start; probe = incr(qf, probe)) {
      if (is_empty_element(get_elem(qf, probe))) {
        uint64_t hi = clrhigh ? 0 : (rand64() & ~mask);
        hash = hi | (probe << qf->qf_rbits) | (rand64() & qf->qf_rmask);
        if (!keys.count(hash)) {
          return hash;
        }
      }
    }
  }

  /* Find a random unused hash. */
  do {
    hash = rand64() & mask;
  } while (keys.count(hash));
  return hash;
}

/* Insert a random p-bit hash into the QF. */
static void ht_put(struct quotient_filter *qf, set<uint64_t> &keys)
{
  uint64_t hash = genhash(qf, true, keys);
  assert(qf_insert(qf, hash));
  keys.insert(hash);
}

/* Remove a hash from the filter. */
static void ht_del(struct quotient_filter *qf, set<uint64_t> &keys)
{
  set<uint64_t>::iterator it;
  uint64_t idx = rand64() % keys.size();
  for (it = keys.begin(); it != keys.end() && idx; ++it, --idx);
  uint64_t hash = *it;
  assert(qf_remove(qf, hash));
  assert(!qf_may_contain(qf, hash));
  keys.erase(hash);
}

/* Check that a set of keys are in the QF. */
static void ht_check(struct quotient_filter *qf, set<uint64_t> &keys)
{
  qf_consistent(qf);
  set<uint64_t>::iterator it;
  for (it = keys.begin(); it != keys.end(); ++it) {
    uint64_t hash = *it;
    assert(qf_may_contain(qf, hash));
  }
}

static void qf_test(struct quotient_filter *qf)
{
  /* Basic get/set tests. */
  uint64_t idx;
  uint64_t size = qf->qf_max_size;
  for (idx = 0; idx < size; ++idx) {
    assert(get_elem(qf, idx) == 0);
    set_elem(qf, idx, idx & qf->qf_elem_mask);
  }
  for (idx = 0; idx < size; ++idx) {
    assert(get_elem(qf, idx) == (idx & qf->qf_elem_mask));
  }
  qf_clear(qf);

  /* Random get/set tests. */
  vector<uint64_t> elements(size, 0);
  for (idx = 0; idx < size; ++idx) {
    uint64_t slot = rand64() % size;
    uint64_t hash = rand64();
    set_elem(qf, slot, hash & qf->qf_elem_mask);
    elements[slot] = hash & qf->qf_elem_mask;
  }
  for (idx = 0; idx < elements.size(); ++idx) {
    assert(get_elem(qf, idx) == elements[idx]);
  }
  qf_clear(qf);

  /* Check: forall x, insert(x) => may-contain(x). */
  set<uint64_t> keys;
  for (idx = 0; idx < size; ++idx) {
    uint64_t elt = genhash(qf, false, keys);
    assert(qf_insert(qf, elt));
    keys.insert(elt);
  }
  ht_check(qf, keys);
  keys.clear();
  qf_clear(qf);

  /* Check that the QF works like a hash set when all keys are p-bit values. */
  for (idx = 0; idx < ROUNDS_MAX; ++idx) {
    while (qf->qf_entries < size) {
      ht_put(qf, keys);
    }

    while (qf->qf_entries > (size / 2)) {
      ht_del(qf, keys);
    }
    ht_check(qf, keys);

    struct qf_iterator qfi;
    qfi_start(qf, &qfi);
    while (!qfi_done(qf, &qfi)) {
      uint64_t hash = qfi_next(qf, &qfi);
      assert(keys.count(hash));
    }
  }
}

/* Fill up the QF (at least partially). */
static void random_fill(struct quotient_filter *qf)
{
  set<uint64_t> keys;
  uint64_t elts = ((uint64_t) rand()) % qf->qf_max_size;
  while (elts) {
    ht_put(qf, keys);
    --elts;
  }
  qf_consistent(qf);
}

/* Check if @lhs is a subset of @rhs. */
static void subsetof(struct quotient_filter *lhs, struct quotient_filter *rhs)
{
  struct qf_iterator qfi;
  qfi_start(lhs, &qfi);
  while (!qfi_done(lhs, &qfi)) {
    uint64_t hash = qfi_next(lhs, &qfi);
    assert(qf_may_contain(rhs, hash));
  }
}

/* Check if @qf contains both @qf1 and @qf2. */
static void supersetof(struct quotient_filter *qf, struct quotient_filter *qf1,
    struct quotient_filter *qf2)
{
  struct qf_iterator qfi;
  qfi_start(qf, &qfi);
  while (!qfi_done(qf, &qfi)) {
    uint64_t hash = qfi_next(qf, &qfi);
    assert(qf_may_contain(qf1, hash) || qf_may_contain(qf2, hash));
  }
}

static void qf_bench()
{
  struct quotient_filter qf;
  const uint32_t q_large = 28;
  const uint32_t q_small = 16;
  const uint32_t nlookups = 1000000;
  struct timeval tv1, tv2;
  uint64_t sec;

  /* Test random inserts + lookups. */
  uint32_t ninserts = (3 * (1 << q_large) / 4);
  printf("Testing %u random inserts and %u lookups", ninserts, nlookups);
  fflush(stdout);
  qf_init(&qf, q_large, 1);
  gettimeofday(&tv1, NULL);
  while (qf.qf_entries < ninserts) {
    assert(qf_insert(&qf, (uint64_t) rand()));
    if (qf.qf_entries % 10000000 == 0) {
      printf(".");
      fflush(stdout);
    }
  }
  for (uint32_t i = 0; i < nlookups; ++i) {
    qf_may_contain(&qf, (uint64_t) rand());
  }
  gettimeofday(&tv2, NULL);
  sec = tv2.tv_sec - tv1.tv_sec;
  printf(" done (%lu seconds).\n", sec);
  fflush(stdout);
  qf_destroy(&qf);

  /* Create a large cluster. Test random lookups. */
  qf_init(&qf, q_small, 1);
  printf("Testing %u contiguous inserts and %u lookups", 1 << q_small,
      nlookups);
  fflush(stdout);
  gettimeofday(&tv1, NULL);
  for (uint64_t quot = 0; quot < (1 << (q_small - 1)); ++quot) {
    uint64_t hash = quot << 1;
    assert(qf_insert(&qf, hash));
    assert(qf_insert(&qf, hash | 1));
    if (quot % 2000 == 0) {
      printf(".");
      fflush(stdout);
    }
  }
  for (uint32_t i = 0; i < nlookups; ++i) {
    qf_may_contain(&qf, (uint64_t) rand());
    if (i % 50000 == 0) {
      printf(".");
      fflush(stdout);
    }
  }
  gettimeofday(&tv2, NULL);
  sec = tv2.tv_sec - tv1.tv_sec;
  printf(" done (%lu seconds).\n", sec);
  fflush(stdout);
  qf_destroy(&qf);
}

int main(int argc, char **argv)
{
  if (argc < 3) {
    fprintf(stderr, "1. log_slot. \n2. load factor. \n3. skewness \n");
    exit(1);
  }
  srand(0);

  uint64_t log_slot = atoi(argv[1]);  /*[CYDBG] Slots*/
  size_t total_items = 1ULL << log_slot;
  size_t max_value = total_items <<8;
  uint64_t load_factor = atoi(argv[2]); /*[CYDBG] load factor*/
  size_t nvals = load_factor * total_items / 100;
	uint64_t skewness = atoi(argv[3]);
	uint64_t *vals;

	if (skewness == 0) {
    vals = (uint64_t*)malloc(total_items*sizeof(vals[0]));
    RAND_bytes((unsigned char*)vals, sizeof(*vals)*total_items);
    for (size_t i = 0; i < total_items; i++) {
       vals[i] %= max_value;
		}
	} else if (skewness > 99) {
	  fprintf(stderr, "Such skewness not allowed\n");
		exit(EXIT_FAILURE);
	} else {
	  uint64_t *foo;
		foo = (uint64_t*)malloc(max_value * sizeof(foo[0]));
		for (uint64_t i = 0; i < max_value; i++) {
		  foo[i] = i;
		}
		unsigned seed;
		seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::shuffle(foo, foo + max_value, std::default_random_engine(seed));
		printf("SHUFFLED\n");

		float skewnessR = skewness / 100.0;
		uint64_t pre_val;
		std::default_random_engine generator;
		vals = (uint64_t*)malloc(nvals * sizeof(vals[0]));
		zipfian_int_distribution<uint64_t> distribution(0, nvals, skewnessR);
		for (uint64_t i = 0; i < nvals; i++) {
		  pre_val = distribution(generator) % max_value;
			vals[i] = foo[pre_val];
		}
	  printf("Zipfian Created\n");
	}

	for (uint64_t i = 0; i < nvals; i++) {
	  printf("%ld\n", vals[i]);
	}


  uint64_t same_num = 0;
  uint64_t *vals_alt = (uint64_t*)malloc(total_items*sizeof(vals_alt[0]));
  RAND_bytes((unsigned char*)vals_alt, sizeof(*vals_alt)*total_items);
  for (size_t i = 0; i < total_items; i++) {
     vals_alt[i] %= max_value;
     for (size_t j = 0; i < total_items; i++) {
       if (vals[j] == vals_alt[i]) {
         same_num++;
       }
     }
  }
  printf("[CYDBG] same_num: %ld\n", same_num);

  struct quotient_filter qf_dummy;
  qf_init(&qf_dummy, log_slot, 8);
  qf_clear(&qf_dummy);
  printf("[CYDBG] size: %ld\n", qf_table_size(log_slot, 8));

  size_t num_inserted = 0;
  struct timeval start, end;
  struct timezone tzp;
  uint64_t elapsed_usecs;

  double insertion_throughput;
  double positive_throughput;
  double negative_throughput;
  double remove_throughput;

  gettimeofday(&start, &tzp);
  for (size_t i = 0; i < nvals; i++, num_inserted++) {
     if (qf_insert(&qf_dummy, vals[i]) == false) {
	break;
     }
  }
  gettimeofday(&end, &tzp);
  elapsed_usecs = tv2usec(&end) - tv2usec(&start);
  insertion_throughput = 1.0 * num_inserted / elapsed_usecs;

  printf("%zu Inserted\n", num_inserted);
  qf_consistent(&qf_dummy);

  printf("Insertion Throughput: %f Million operations / sec\n", insertion_throughput);

  gettimeofday(&start, &tzp);
  for (size_t i = 0; i < num_inserted; i++) {
     assert(qf_may_contain(&qf_dummy, vals[i]));
  }
  gettimeofday(&end, &tzp);
  elapsed_usecs = tv2usec(&end) - tv2usec(&start);
  positive_throughput = 1.0 * num_inserted / elapsed_usecs;

  printf("Positive Throughput: %f Million operations / sec\n", positive_throughput);

  gettimeofday(&start, &tzp);
  size_t total_queries = 0; 
  size_t false_queries = 0;
  for (size_t i = 0; i < num_inserted; i++){
     if (qf_may_contain(&qf_dummy, vals_alt[i]) == true) {
	false_queries++;
     }
     total_queries++;
  }
  gettimeofday(&end, &tzp);
  elapsed_usecs = tv2usec(&end) - tv2usec(&start);
  negative_throughput = 1.0 * num_inserted / elapsed_usecs;

  printf("Negative Throughput: %f Million operations / sec\n", negative_throughput);
  printf("false_queries: %ld, total_queries: %ld\n", false_queries, total_queries);

  gettimeofday(&start, &tzp);
  size_t num_removed = num_inserted;
  for (size_t i = 0; i < num_inserted; i++) {
     if (qf_remove(&qf_dummy, vals[i]) == false) {
	num_removed--;
     }
  }
  gettimeofday(&end, &tzp);
  elapsed_usecs = tv2usec(&end) - tv2usec(&start);
  remove_throughput = 1.0 * num_removed / elapsed_usecs;

  printf("Remove Throughput: %f Million operations / sec\n", remove_throughput);

  qf_destroy(&qf_dummy);

  /*
#if QBENCH
  qf_bench();
#else
  for (uint32_t q = 1; q <= Q_MAX; ++q) {
    printf("Starting rounds for qf_test::q=%u\n", q);

#pragma omp parallel for
    for (uint32_t r = 1; r <= R_MAX; ++r) {
      struct quotient_filter qf;
      if (!qf_init(&qf, q, r)) {
        fail(&qf, "init-1");
      }
      qf_test(&qf);
      qf_destroy(&qf);
    }
  }

  for (uint32_t q1 = 1; q1 <= Q_MAX; ++q1) {
    for (uint32_t r1 = 1; r1 <= R_MAX; ++r1) {
      for (uint32_t q2 = 1; q2 <= Q_MAX; ++q2) {
        printf("Starting rounds for qf_merge::q1=%u,q2=%u\n", q1, q2);

#pragma omp parallel for
        for (uint32_t r2 = 1; r2 <= R_MAX; ++r2) {
          struct quotient_filter qf;
          struct quotient_filter qf1, qf2;
          if (!qf_init(&qf1, q1, r1) || !qf_init(&qf2, q2, r2)) {
            fail(&qf1, "init-2");
          }

          random_fill(&qf1);
          random_fill(&qf2);
          assert(qf_merge(&qf1, &qf2, &qf));
          qf_consistent(&qf);
          subsetof(&qf1, &qf);
          subsetof(&qf2, &qf);
          supersetof(&qf, &qf1, &qf2);
          qf_destroy(&qf1);
          qf_destroy(&qf2);
          qf_destroy(&qf);
        }
      }
    }
  }
#endif // QBENCH
  */

  puts("[PASSED] qf tests");
  return 0;
}
