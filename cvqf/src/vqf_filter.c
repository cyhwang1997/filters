/*
 * ============================================================================
 *
 *       Filename:  vqf_filter.c
 *
 *         Author:  Prashant Pandey (), ppandey@berkeley.edu
 *   Organization:  LBNL/UCB
 *
 * ============================================================================
 */

#include <algorithm>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>  // portable to all x86 compilers
#include <tmmintrin.h>

#include "vqf_filter.h"
#include "vqf_precompute.h"

// ALT block check is set of 75% of the number of slots
#if TAG_BITS == 8
#define TAG_MASK 0xff
#define QUQU_SLOTS_PER_BLOCK 48
#define QUQU_BUCKETS_PER_BLOCK 80
#define QUQU_CHECK_ALT 92

// additional
#define QUQU_MAX 255
#define QUQU_PRESLOT 16

#elif TAG_BITS == 12
#define TAG_MASK 0xfff
#define QUQU_SLOTS_PER_BLOCK 32
#define QUQU_BUCKETS_PER_BLOCK 96
#define QUQU_CHECK_ALT 104

#elif TAG_BITS == 16
#define TAG_MASK 0xffff
#define QUQU_SLOTS_PER_BLOCK 28 
#define QUQU_BUCKETS_PER_BLOCK 36
#define QUQU_CHECK_ALT 43

// additional
#define QUQU_MAX 65535
#define QUQU_PRESLOT 4

#endif

#ifdef __AVX512BW__
extern __m512i SHUFFLE [];
extern __m512i SHUFFLE_REMOVE [];
extern __m512i SHUFFLE16 [];
extern __m512i SHUFFLE_REMOVE16 [];
#endif

#define LOCK_MASK (1ULL << 63)
#define UNLOCK_MASK ~(1ULL << 63)

static inline bool check_tags(vqf_filter * restrict filter, uint64_t tag, uint64_t block_index);

static inline void lock(vqf_block& block)
{
#ifdef ENABLE_THREADS
   uint64_t *data;
#if TAG_BITS == 8
   data = block.md + 1;
#elif TAG_BITS == 16
   data = &block.md;
#endif
   while ((__sync_fetch_and_or(data, LOCK_MASK) & (1ULL << 63)) != 0) {}
#endif
}

static inline void unlock(vqf_block& block)
{
#ifdef ENABLE_THREADS
   uint64_t *data;
#if TAG_BITS == 8
   data = block.md + 1;
#elif TAG_BITS == 16
   data = &block.md;
#endif
   __sync_fetch_and_and(data, UNLOCK_MASK);
#endif
}

static inline void lock_blocks(vqf_filter * restrict filter, uint64_t index1, uint64_t index2)  {
#ifdef ENABLE_THREADS
   if (index1 < index2) {
      lock(filter->blocks[index1/QUQU_BUCKETS_PER_BLOCK].block);
      lock(filter->blocks[index2/QUQU_BUCKETS_PER_BLOCK].block);
   } else {
      lock(filter->blocks[index2/QUQU_BUCKETS_PER_BLOCK].block);
      lock(filter->blocks[index1/QUQU_BUCKETS_PER_BLOCK].block);
   }
#endif
}

static inline void unlock_blocks(vqf_filter * restrict filter, uint64_t index1, uint64_t index2)  {
#ifdef ENABLE_THREADS
   if (index1 < index2) {
      unlock(filter->blocks[index1/QUQU_BUCKETS_PER_BLOCK].block);
      unlock(filter->blocks[index2/QUQU_BUCKETS_PER_BLOCK].block);
   } else {
      unlock(filter->blocks[index2/QUQU_BUCKETS_PER_BLOCK].block);
      unlock(filter->blocks[index1/QUQU_BUCKETS_PER_BLOCK].block);
   }
#endif
}

static inline int word_rank(uint64_t val) {
   return __builtin_popcountll(val);
}

// Returns the position of the rank'th 1.  (rank = 0 returns the 1st 1)
// Returns 64 if there are fewer than rank+1 1s.
static inline uint64_t word_select(uint64_t val, int rank) {
   val = _pdep_u64(one[rank], val);
   return _tzcnt_u64(val);
}

// select(vec, 0) -> -1
// select(vec, i) -> 128, if i > popcnt(vec)
static inline int64_t select_128_old(__uint128_t vector, uint64_t rank) {
   uint64_t lower_word = vector & 0xffffffffffffffff;
   uint64_t lower_pdep = _pdep_u64(one[rank], lower_word);
   //uint64_t lower_select = word_select(lower_word, rank);
   if (lower_pdep != 0) {
      //assert(rank < word_rank(lower_word));
      return _tzcnt_u64(lower_pdep);
   }
   rank = rank - word_rank(lower_word);
   uint64_t higher_word = vector >> 64;
   return word_select(higher_word, rank) + 64;
}

static inline uint64_t lookup_64(uint64_t vector, uint64_t rank) {
   uint64_t lower_return = _pdep_u64(one[rank], vector) >> rank << (sizeof(uint64_t)/2);
   return lower_return;
}

static inline uint64_t lookup_128(uint64_t *vector, uint64_t rank) {
   uint64_t lower_word = vector[0];
   uint64_t lower_rank = word_rank(lower_word);
   uint64_t lower_return = _pdep_u64(one[rank], lower_word) >> rank << sizeof(__uint128_t);
   int64_t higher_rank = (int64_t)rank - lower_rank;
   uint64_t higher_word = vector[1];
   uint64_t higher_return = _pdep_u64(one[higher_rank], higher_word);
   higher_return <<= (64 + sizeof(__uint128_t) - rank);
   return lower_return + higher_return;
}

static inline int64_t select_64(uint64_t vector, uint64_t rank) {
   return _tzcnt_u64(lookup_64(vector, rank));
}

static inline int64_t select_128(uint64_t *vector, uint64_t rank) {
   return _tzcnt_u64(lookup_128(vector, rank));
}

void print_m512i(__m512i vec) {
  uint8_t buffer[64];
  _mm512_storeu_si512(buffer, vec);

  printf("print_m512i: ");
  for (int i = 0; i < 64; i++) {
    printf("%x ", buffer[i]);
  }
  printf("\n");
}

//assumes little endian
#if TAG_BITS == 8
void print_bits(__uint128_t num, int numbits)
{
   int i;
   for (i = 0 ; i < numbits; i++) {
      if (i != 0 && i % 8 == 0) {
         printf(":");
      }
      printf("%d", ((num >> i) & 1) == 1);
   }
   puts("");
}

void print_tags(uint8_t *tags, uint32_t size) {
   for (uint8_t i = 0; i < size; i++)
      printf("%d ", (uint32_t)tags[i]);
   printf("\n");
}

void print_tags_special(uint8_t *tags, uint32_t size, uint64_t slot_index) {
   for (uint8_t i = 0; i < size; i++) {
      if (i == slot_index) {
	 printf("[%d] ", (uint32_t)tags[i]);
      }

      else {
	 printf("%d ", (uint32_t)tags[i]);
      }
   }
   printf("\n");
}

void print_block(vqf_filter *filter, uint64_t block_index) {
   printf("block index: %ld\n", block_index);
   printf("metadata: ");
   uint64_t *md = filter->blocks[block_index].block.md;
   print_bits(*(__uint128_t *)md, QUQU_BUCKETS_PER_BLOCK +
         QUQU_SLOTS_PER_BLOCK);
   printf("tags: ");
   print_tags(filter->blocks[block_index].block.tags, QUQU_SLOTS_PER_BLOCK);
}
/*CY*/
void print_next_block(vqf_filter *filter, uint64_t block_index) {
   printf("block index: %ld\n", block_index);
   printf("metadata: ");
   uint64_t *md = filter->blocks[block_index].next->block.md;
   print_bits(*(__uint128_t *)md, QUQU_BUCKETS_PER_BLOCK +
         QUQU_SLOTS_PER_BLOCK);
   printf("tags: ");
   print_tags(filter->blocks[block_index].next->block.tags, QUQU_SLOTS_PER_BLOCK);
}
/*CY*/
void print_block_special(vqf_filter *filter, uint64_t block_index, uint64_t slot_index) {
   printf("block index: %ld\n", block_index);
   printf("metadata: ");
   uint64_t *md = filter->blocks[block_index].block.md;
   print_bits(*(__uint128_t *)md, QUQU_BUCKETS_PER_BLOCK +
         QUQU_SLOTS_PER_BLOCK);
   printf("tags: ");
   print_tags_special(filter->blocks[block_index].block.tags, QUQU_SLOTS_PER_BLOCK, slot_index);
}
#elif TAG_BITS == 16
void print_bits(uint64_t num, int numbits)
{
   int i;
   for (i = 0 ; i < numbits; i++) {
      if (i != 0 && i % 8 == 0) {
         printf(":");
      }
      printf("%d", ((num >> i) & 1) == 1);
   }
   puts("");
}
void print_tags(uint16_t *tags, uint32_t size) {
   for (uint8_t i = 0; i < size; i++)
      printf("%d ", (uint32_t)tags[i]);
   printf("\n");
}
void print_block(vqf_filter *filter, uint64_t block_index) {
   printf("block index: %ld\n", block_index);
   printf("metadata: ");
   uint64_t md = filter->blocks[block_index].block.md;
   print_bits(md, QUQU_BUCKETS_PER_BLOCK + QUQU_SLOTS_PER_BLOCK);
   printf("tags: ");
   print_tags(filter->blocks[block_index].block.tags, QUQU_SLOTS_PER_BLOCK);
}
#endif

#ifdef __AVX512BW__
#if TAG_BITS == 8
static inline void update_tags_512(vqf_block * restrict block, uint8_t index, uint8_t tag) {
   block->tags[47] = tag;	// add tag at the end

   __m512i vector = _mm512_loadu_si512(reinterpret_cast<__m512i*>(block));
   vector = _mm512_permutexvar_epi8(SHUFFLE[index], vector);
   _mm512_storeu_si512(reinterpret_cast<__m512i*>(block), vector);
}

static inline void remove_tags_512(vqf_block * restrict block, uint8_t index) {
   __m512i vector = _mm512_loadu_si512(reinterpret_cast<__m512i*>(block));
   vector = _mm512_permutexvar_epi8(SHUFFLE_REMOVE[index], vector);
   _mm512_storeu_si512(reinterpret_cast<__m512i*>(block), vector);
}
#elif TAG_BITS == 16
static inline void update_tags_512(vqf_block * restrict block, uint8_t index, uint16_t tag) {
   block->tags[27] = tag;	// add tag at the end

   __m512i vector = _mm512_loadu_si512(reinterpret_cast<__m512i*>(block));
   vector = _mm512_permutexvar_epi16(SHUFFLE16[index], vector);
   _mm512_storeu_si512(reinterpret_cast<__m512i*>(block), vector);
}

static inline void remove_tags_512(vqf_block * restrict block, uint8_t index) {
   __m512i vector = _mm512_loadu_si512(reinterpret_cast<__m512i*>(block));
   vector = _mm512_permutexvar_epi16(SHUFFLE_REMOVE16[index], vector);
   _mm512_storeu_si512(reinterpret_cast<__m512i*>(block), vector);
}
#endif
#else
#if TAG_BITS == 8
static inline void update_tags_512(vqf_block * restrict block, uint8_t index, uint8_t tag) {
   index -= 16;
   memmove(&block->tags[index + 1], &block->tags[index], sizeof(block->tags) / sizeof(block->tags[0]) - index - 1);
   block->tags[index] = tag;
}

static inline void remove_tags_512(vqf_block * restrict block, uint8_t index) {
   index -= 16;
   memmove(&block->tags[index], &block->tags[index+1], sizeof(block->tags) / sizeof(block->tags[0]) - index);
}
#elif TAG_BITS == 16
static inline void update_tags_512(vqf_block * restrict block, uint8_t index, uint16_t tag) {
   index -= 4;
   memmove(&block->tags[index + 1], &block->tags[index], (sizeof(block->tags) / sizeof(block->tags[0]) - index - 1) * 2);
   block->tags[index] = tag;
}

static inline void remove_tags_512(vqf_block * restrict block, uint8_t index) {
   index -= 4;
   memmove(&block->tags[index], &block->tags[index+1], (sizeof(block->tags) / sizeof(block->tags[0]) - index) * 2);
}
#endif
#endif

#if 0
// Shuffle using AVX2 vector instruction. It turns out memmove is faster compared to AVX2.
inline __m256i cross_lane_shuffle(const __m256i & value, const __m256i &
      shuffle) 
{ 
   return _mm256_or_si256(_mm256_shuffle_epi8(value, _mm256_add_epi8(shuffle,
               K[0])), 
         _mm256_shuffle_epi8(_mm256_permute4x64_epi64(value, 0x4E),
            _mm256_add_epi8(shuffle, K[1]))); 
} 

#define SHUFFLE_SIZE 32
void shuffle_256(uint8_t * restrict source, __m256i shuffle) {
   __m256i vector = _mm256_loadu_si256(reinterpret_cast<__m256i*>(source)); 

   vector = cross_lane_shuffle(vector, shuffle); 
   _mm256_storeu_si256(reinterpret_cast<__m256i*>(source), vector); 
} 

static inline void update_tags_256(uint8_t * restrict block, uint8_t index,
      uint8_t tag) {
   index = index + sizeof(__uint128_t);	// offset index based on md field.
   block[63] = tag;	// add tag at the end
   shuffle_256(block + SHUFFLE_SIZE, RM[index]); // right block shuffle
   if (index < SHUFFLE_SIZE) {		// if index lies in the left block
      std::swap(block[31], block[32]);	// move tag to the end of left block
      shuffle_256(block, LM[index]);	// shuffle left block
   }
}
#endif

#if TAG_BITS == 8
static inline void update_md(uint64_t *md, uint8_t index) {
   uint64_t carry = (md[0] >> 63) & carry_pdep_table[index];
   md[1] = _pdep_u64(md[1],         high_order_pdep_table[index]) | carry;
   md[0] = _pdep_u64(md[0],         low_order_pdep_table[index]);
}

static inline void remove_md(uint64_t *md, uint8_t index) {
   uint64_t carry = (md[1] & carry_pdep_table[index]) << 63;
   md[1] = _pext_u64(md[1],  high_order_pdep_table[index]) | (1ULL << 63);
   md[0] = _pext_u64(md[0],  low_order_pdep_table[index]) | carry;
}

// number of 0s in the metadata is the number of tags.
static inline uint64_t get_block_free_space(uint64_t *vector) {
   uint64_t lower_word = vector[0];
   uint64_t higher_word = vector[1];
   return word_rank(lower_word) + word_rank(higher_word);
}
#elif TAG_BITS == 16
static inline void update_md(uint64_t *md, uint8_t index) {
   *md = _pdep_u64(*md, low_order_pdep_table[index]);
}

static inline void remove_md(uint64_t *md, uint8_t index) {
   *md = _pext_u64(*md, low_order_pdep_table[index]) | (1ULL << 63);
}

// number of 0s in the metadata is the number of tags.
static inline uint64_t get_block_free_space(uint64_t vector) {
   return word_rank(vector);
}
#endif

// Create n/log(n) blocks of log(n) slots.
// log(n) is 51 given a cache line size.
// n/51 blocks.
vqf_filter * vqf_init(uint64_t nslots) {
   vqf_filter *filter;

   uint64_t total_blocks = (nslots + QUQU_SLOTS_PER_BLOCK)/QUQU_SLOTS_PER_BLOCK;
//   uint64_t total_size_in_bytes = sizeof(vqf_block) * total_blocks;
   uint64_t total_size_in_bytes = sizeof(linked_blocks) * total_blocks;

   //filter = (vqf_filter *)malloc(sizeof(*filter));
   filter = (vqf_filter *)malloc(sizeof(*filter) + total_size_in_bytes);
   printf("Size: %ld, total_blocks: %ld\n",total_size_in_bytes, total_blocks);
   assert(filter);

   filter->metadata.total_size_in_bytes = total_size_in_bytes;
   filter->metadata.nslots = total_blocks * QUQU_SLOTS_PER_BLOCK;
#if TAG_BITS == 8
   filter->metadata.key_remainder_bits = 8;
#elif TAG_BITS == 16
   filter->metadata.key_remainder_bits = 16;
#endif
   filter->metadata.range = total_blocks * QUQU_BUCKETS_PER_BLOCK * (1ULL << filter->metadata.key_remainder_bits);
   filter->metadata.nblocks = total_blocks;
   filter->metadata.nelts = 0;

   // memset to 1
#if TAG_BITS == 8
   for (uint64_t i = 0; i < total_blocks; i++) {
      filter->blocks[i].block.md[0] = UINT64_MAX;
      filter->blocks[i].block.md[1] = UINT64_MAX;
      filter->blocks[i].next = NULL;
      // reset the most significant bit of metadata for locking.
//[CYDBG] commented out for single thread      filter->blocks[i].md[1] = filter->blocks[i].md[1] & ~(1ULL << 63);
   }
#elif TAG_BITS == 16
   for (uint64_t i = 0; i < total_blocks; i++) {
      filter->blocks[i].block.md = UINT64_MAX;
      filter->blocks[i].block.md = filter->blocks[i].block.md & ~(1ULL << 63);
      filter->blocks[i].next = NULL;
   }
#endif

   return filter;
}

// If the item goes in the i'th slot (starting from 0) in the block then
// find the i'th 0 in the metadata, insert a 1 after that and shift the rest
// by 1 bit.
// Insert the new tag at the end of its run and shift the rest by 1 slot.
int vqf_insert(vqf_filter * restrict filter, uint64_t hash) { // bool
   vqf_metadata * restrict metadata           = &filter->metadata;
//   vqf_block    * restrict blocks             = filter->blocks;
   linked_blocks    * restrict blocks             = filter->blocks;
   uint64_t                 key_remainder_bits = metadata->key_remainder_bits;
   uint64_t                 range              = metadata->range;


   uint64_t block_index = hash >> key_remainder_bits;
   /*CY*/
   linked_blocks *last_block = &blocks[block_index/QUQU_BUCKETS_PER_BLOCK];
   while (last_block->next != NULL) {
     last_block = last_block->next;
   }
   vqf_block * cur_block = &last_block->block;
   /*CY*/
   lock(*cur_block);
#if TAG_BITS == 8
   uint64_t *block_md = cur_block->md;
   uint64_t block_free = get_block_free_space(block_md);
#elif TAG_BITS == 16
   uint64_t *block_md = &cur_block->md;
   uint64_t block_free = get_block_free_space(*block_md);
#endif
   uint64_t tag = hash & TAG_MASK;
   uint64_t alt_block_index = ((hash ^ (tag * 0x5bd1e995)) % range) >> key_remainder_bits;
   uint64_t alt_ = ((hash >> 8) ^ ((tag * 0x5bd1e995) >> 8)) % (range >> 8);
   if (alt_ != alt_block_index) {
     printf("[CYDBG] Different!\n");
   }

   linked_blocks *last_alt_block = &blocks[alt_block_index/QUQU_BUCKETS_PER_BLOCK];
   while(last_alt_block->next != NULL) {
     printf("alt_block next block exists\n");
     last_alt_block = last_alt_block->next;
   }
   vqf_block * cur_alt_block = &last_alt_block->block;

   __builtin_prefetch(&blocks[alt_block_index/QUQU_BUCKETS_PER_BLOCK].block);

   // block_index is over 75% full && the two candidate blocks are different
   if (block_free < QUQU_CHECK_ALT && block_index/QUQU_BUCKETS_PER_BLOCK != alt_block_index/QUQU_BUCKETS_PER_BLOCK) {
      unlock(*cur_block);
      lock_blocks(filter, block_index, alt_block_index); //CY: needs to change
#if TAG_BITS == 8
      uint64_t *alt_block_md = cur_alt_block->md;
      uint64_t alt_block_free = get_block_free_space(alt_block_md);
#elif TAG_BITS == 16
      uint64_t *alt_block_md = &cur_alt_block->md;
      uint64_t alt_block_free = get_block_free_space(*alt_block_md);
#endif
      // pick the least loaded block
      if (alt_block_free > block_free) {
        unlock(*cur_block);
        block_index = alt_block_index;
        cur_block = cur_alt_block;
        block_md = alt_block_md;
        block_free = alt_block_free;
      } else if (block_free == QUQU_BUCKETS_PER_BLOCK) {
        if (check_space(filter, tag, block_index)) {
          unlock(*cur_alt_block);
        } else if (check_space(filter, tag, alt_block_index)) {
          unlock(blocks[block_index/QUQU_BUCKETS_PER_BLOCK].block);
          block_index = alt_block_index;
          cur_block = cur_alt_block;
          block_md = alt_block_md;
          block_free = alt_block_free;
        } else { //both blocks are full
          print_block(filter, block_index / QUQU_BUCKETS_PER_BLOCK);
          print_block(filter, alt_block_index / QUQU_BUCKETS_PER_BLOCK);
          unlock_blocks(filter, block_index, alt_block_index);
          cur_block = add_block(filter, block_index / QUQU_BUCKETS_PER_BLOCK);
/*          printf("--------both blocks are full in vqf_insert--------\n");
          printf("block index: %ld\n", block_index / QUQU_BUCKETS_PER_BLOCK);
          printf("metadata: ");
          print_bits(*(__uint128_t *)cur_block->md, QUQU_BUCKETS_PER_BLOCK + QUQU_SLOTS_PER_BLOCK);
          printf("tags: ");
          print_tags(cur_block->tags, QUQU_SLOTS_PER_BLOCK);
          printf("[CYDBG] hash: %ld, tag: %ld\n", hash, tag);*/
          fprintf(stderr, "vqf filter is full.\n");
          return -1;
        }
      } else {
        unlock(*cur_block);
      }
   } else if (block_index/QUQU_BUCKETS_PER_BLOCK == alt_block_index/QUQU_BUCKETS_PER_BLOCK) {
     if (block_free == QUQU_BUCKETS_PER_BLOCK) {
       print_block(filter, block_index / QUQU_BUCKETS_PER_BLOCK);
       print_block(filter, alt_block_index / QUQU_BUCKETS_PER_BLOCK);
       unlock_blocks(filter, block_index, alt_block_index);
       cur_block = add_block(filter, block_index / QUQU_BUCKETS_PER_BLOCK);
       printf("[CYDBG] hash: %ld, tag: %ld\n", hash, tag);
       fprintf(stderr, "vqf filter is full, no alternative block\n");
       return -1;
     } else {
       unlock(blocks[alt_block_index/QUQU_BUCKETS_PER_BLOCK].block);
       cur_block = &blocks[block_index/QUQU_BUCKETS_PER_BLOCK].block;
     }
   }

//   uint64_t index = block_index / QUQU_BUCKETS_PER_BLOCK;
   uint64_t offset = block_index % QUQU_BUCKETS_PER_BLOCK;

//   cur_block = &blocks[index].block;
   uint64_t *cur_md = cur_block->md;

#if TAG_BITS == 8
   uint64_t slot_index = select_128(cur_md, offset);
   uint64_t select_index = slot_index + offset - sizeof(__uint128_t);

   /*CVQF*/
   uint64_t preslot_index; // yongjin
   if (offset != 0) {
     preslot_index = select_128(cur_md, offset - 1);
   } else {
     preslot_index = QUQU_PRESLOT;
   }
   /*CVQF*/
#elif TAG_BITS == 16
   uint64_t slot_index = select_64(*cur_md, offset);
   uint64_t select_index = slot_index + offset - (sizeof(uint64_t)/2);
   uint64_t preslot_index; // yongjin
   if (offset != 0) {
     preslot_index = select_128(cur_md, offset - 1);
   } else {
     preslot_index = QUQU_PRESLOT;
   }
#endif
//   printf("[CYDBG] hash: %lx, offset: %lu, tag: %lu\n", hash, offset, tag);

   /*CVQF*/
   uint64_t target_index;
   uint64_t end_target_index;
   uint8_t temp_tag;

   // bucket is empty
   if (preslot_index == slot_index) {
     target_index = slot_index;
     update_tags_512(cur_block, target_index, tag);
     update_md(cur_md, select_index);
     unlock(*cur_block);
//     print_block(filter, index);
     //return true;
     return 1;
   }

   // bucket is not empty
   else {
     target_index = slot_index;
     // sorting ////////////////// find the position(target_index) to insert. the tags in a bucket are sorted in ascending order
     for (uint64_t i = preslot_index; i < slot_index; i++) {
       // candidate
       if (cur_block->tags[i - QUQU_PRESLOT] >= tag) {
         // the first tag is bigger than the current tag. insert it to the first slot
         if (i == preslot_index) {
           target_index = i;
           break;
         }
         // could be counter
         else if (cur_block->tags[i - 1 - QUQU_PRESLOT] == 0) {
           // CY: the first slot is '0' and the current slot is a counter of tag '0'
           // inserting 0
           if (i == preslot_index + 1 && tag == 0) {
             /*CY*/
             while (cur_block->tags[i - QUQU_PRESLOT] == QUQU_MAX) 
               i++;
             if (cur_block->tags[i - QUQU_PRESLOT] != 0)
               i += 2; /*[0, 2, 0, 0], current tag is 2*/
             else
               i++;  /*[0, 255, 0, 0], current tag is 0 right after 255*/
           }
           // The current index is not a counter. The '0' before this current index is from tag '0'
           else if (cur_block->tags[i - 2 - QUQU_PRESLOT] == 0) { 
             target_index = i;
             break;
           }
           // CY: 0 is inserted only once, and the current index inserted right after '0'
           else if (cur_block->tags[preslot_index - QUQU_PRESLOT] == 0 && i == preslot_index + 1) {
             target_index = i;
             break;
           }

           // other cases
           else {
             temp_tag = cur_block->tags[i - 2 - QUQU_PRESLOT];
             while(cur_block->tags[i - QUQU_PRESLOT] == QUQU_MAX) 
               i++;
             if (cur_block->tags[i - QUQU_PRESLOT] != temp_tag) 
               i++;
             continue;
           }
         }

         // found
         else {
           target_index = i;
           break;
         }
       }
     }
   }
   // sorting //////////////////
//   printf("[CYDBG] slot_index: %ld, target_index: %ld\n", slot_index, target_index);
//   print_block(filter, index);

   // if tag that is ">=" is found in [preslot_index ---------- slot_index)
   if (target_index < slot_index) {

     // need counter
     if (cur_block->tags[target_index - QUQU_PRESLOT] == tag) {

       // just find the other tag in [preslot_index ---------- slot_index)
       end_target_index = target_index + 1;
       while (end_target_index < slot_index) {

	 // (if end_target_index == slot_index, there is no match)
         if (cur_block->tags[end_target_index - QUQU_PRESLOT] == tag) break;
           end_target_index++;
         }

       // no extra match, just put it
         if (end_target_index == slot_index) {
           update_tags_512(cur_block, target_index, tag);
           update_md(cur_md, select_index);
           unlock(*cur_block);
//           print_block(filter, index);
           //return true;
           return 1;
       }

       // counter //////////////////

       // extra match, tag is 0
       else if (tag == 0) {

         // [0, 0, ...]
         if (end_target_index == target_index + 1) {

           // check if [0, 0, 0, ...]
           if (end_target_index < slot_index - 1) {
             if (cur_block->tags[end_target_index + 1 - QUQU_PRESLOT] == tag) {
               update_tags_512(cur_block, end_target_index, 1);
               update_md(cur_md, select_index);
               unlock(*cur_block);
//               print_block(filter, index);
               return 1;
             }

             else {
               update_tags_512(cur_block, end_target_index, tag);
               update_md(cur_md, select_index);
               unlock(*cur_block);
//               print_block(filter, index);
               return 1;
             }
	         }

           // [0, 0]
           else {
             update_tags_512(cur_block, target_index, tag);
             update_md(cur_md, select_index);
             unlock(*cur_block);
//             print_block(filter, index);
             return 1;
	         }
         }

         // check if counter, [0, ... 0, 0, ...]
         else if (end_target_index < slot_index - 1) {
           if (cur_block->tags[end_target_index + 1 - QUQU_PRESLOT] == tag) {

             // full counter
             if (cur_block->tags[end_target_index - 1 - QUQU_PRESLOT] == QUQU_MAX) {
               update_tags_512(cur_block, end_target_index, 1);
               update_md(cur_md, select_index);
               unlock(*cur_block);
//               print_block(filter, index);
               return 1;
             }

             // increment counter
             else {
               cur_block->tags[end_target_index - 1 - QUQU_PRESLOT]++;
//               print_block(filter, index);
               unlock(*cur_block);
               return 0;
             }
           }

           else {
             // wrong zero fetched
             update_tags_512(cur_block, target_index, tag);
             update_md(cur_md, select_index);
//             print_block(filter, index);
             unlock(*cur_block);
             return 1;
           }
         }

         // wrong extra 0 fetched
         else {
           // [0 ... 0] ?
           printf("ERROR\n");
           //update_tags_512(&blocks[index], target_index, tag);
           //update_md(cur_md, select_index);
           unlock(*cur_block);
           //return false;
           return -1;
         }
       }

       // extra match, tag is 1
       else if (tag == 1) {

         // [1, 1], need to insert two tags
         if (end_target_index == target_index + 1) {

           // cannot insert two tags
           if (block_free == QUQU_BUCKETS_PER_BLOCK + 1) {
             unlock(*cur_block);
             //return false;
             return -1;
           }

           // can insert two tags
           else {
             update_tags_512(cur_block, end_target_index, 0);
             update_md(cur_md, select_index);
             update_tags_512(cur_block, end_target_index + 1, 2);
             update_md(cur_md, select_index);
//             print_block(filter, index);
             unlock(*cur_block);
             //return true;
             return 2;
           }
         }

         // counter
         else if (cur_block->tags[target_index + 1 - QUQU_PRESLOT] < tag) {

           // add new counter
           if (cur_block->tags[end_target_index - 1  - QUQU_PRESLOT] == QUQU_MAX) {
             update_tags_512(cur_block, end_target_index, 2);
             update_md(cur_md, select_index);
//             print_block(filter, index);
             unlock(*cur_block);
             //return true;
             return 1;
           }

           // increment counter
           else {
             cur_block->tags[end_target_index - 1 - QUQU_PRESLOT]++;
//             print_block(filter, index);
             unlock(*cur_block);
             //return true;
             return 0;
           }
         }

         // wrong extra 1 fetched
         else {
           update_tags_512(cur_block, target_index, tag);
           update_md(cur_md, select_index);
//           print_block(filter, index);
           unlock(*cur_block);
           //unlock(blocks[block_index/QUQU_BUCKETS_PER_BLOCK]);
           //return true;
           return 1;
         }
       }

       // extra match, tag is 255
       else if (tag == QUQU_MAX) {

         // [255, 255]
         if (end_target_index == target_index + 1) {
           update_tags_512(cur_block, end_target_index, 1);
           update_md(cur_md, select_index);
//           print_block(filter, index);
           unlock(*cur_block);
           //return true;
           return 1;
         }

         // [255, ... , 255]
         else {

           // add new counter
           if (cur_block->tags[end_target_index - 1 - QUQU_PRESLOT] == QUQU_MAX - 1) {
             update_tags_512(cur_block, end_target_index, 1);
             update_md(cur_md, select_index);
//             print_block(filter, index);
             unlock(*cur_block);
             //return true;
             return 1;
           }

           // increment counter
           else {
             cur_block->tags[end_target_index - 1 - QUQU_PRESLOT]++;
//             print_block(filter, index);
             unlock(*cur_block);
             //return true;
             return 0;
           }
         }
       }

       // extra match, other tags
       else {
         // [tag, tag]
         if (end_target_index == target_index + 1) {
           update_tags_512(cur_block, end_target_index, 1);
           update_md(cur_md, select_index);
//           print_block(filter, index);
           unlock(*cur_block);
           //return true;
           return 1;
         }

         // counter
         else if (cur_block->tags[target_index + 1 - QUQU_PRESLOT] < tag) {
           // add new counter
           if (cur_block->tags[end_target_index - 1 - QUQU_PRESLOT] == QUQU_MAX) {
             update_tags_512(cur_block, end_target_index, 1);
             update_md(cur_md, select_index);
//               print_block(filter, index);
             unlock(*cur_block);
             //return true;
             return 1;
           }

           // increment counter
           else {
             temp_tag = cur_block->tags[end_target_index - 1 - QUQU_PRESLOT] + 1;
             if (temp_tag == tag) {
               temp_tag++;

               // need to put 0
               if (cur_block->tags[target_index + 1 - QUQU_PRESLOT] != 0) {
                 cur_block->tags[end_target_index - 1 - QUQU_PRESLOT] = temp_tag;
                 update_tags_512(cur_block, target_index + 1, 0);
                 update_md(cur_md, select_index);
//                 print_block(filter, index);
                 unlock(*cur_block);
                 //return true;
                 return 1;
               }

               // no need to put 0
               else {
                 cur_block->tags[end_target_index - 1 - QUQU_PRESLOT] = temp_tag;
//                 print_block(filter, index);
                 unlock(*cur_block);
                 //return true;
                 return 0;
               }
             } else {
/*               if (hash == 27814299785) {
                 printf("[CYDBG] FF\n");
                 printf("[CYDBG] %d, %d\n", blocks[index].tags[end_target_index - 1 - QUQU_PRESLOT], temp_tag);
                 printf("[CYDBG] end_target_index: %ld\n", end_target_index);
               }*/
/*               if (hash == 27814299785) 
                 print_block(filter, 1358120);*/
               cur_block->tags[end_target_index - 1 - QUQU_PRESLOT] = temp_tag;
//               print_block(filter, index);
               unlock(*cur_block);
/*               if (hash == 27814299785) 
                 print_block(filter, 1358120);*/
               //return true;
               return 0;
             }
           }
         }

         // wrong fetch
         else {
           update_tags_512(cur_block, target_index, tag);
           update_md(cur_md, select_index);
//           print_block(filter, index);
           unlock(*cur_block);
           //return true;
           return 1;
         }
       }
       // counter //////////////////
     }

     // no need counter
     else {
       update_tags_512(cur_block, target_index, tag);
       update_md(cur_md, select_index);
//       print_block(filter, index);
       unlock(*cur_block);
       //unlock(blocks[block_index/QUQU_BUCKETS_PER_BLOCK]);
       //return true;
       return 1;
     }
   }

   // if not found in [preslot_index ---------- slot_index)
   else {
     update_tags_512(cur_block, target_index, tag); // slot_index
     update_md(cur_md, select_index);
//     print_block(filter, index);
     unlock(*cur_block);
     //unlock(blocks[block_index/QUQU_BUCKETS_PER_BLOCK]);
     //return true;
     return 1;
   }

   //print_block(filter, index);
   // something wrong
   return false;

   /*update_tags_512(&blocks[index], target_index, tag);
   update_md(block_md, select_index);
   unlock(blocks[index]);
   return true;*/
}

static inline bool remove_tags(vqf_filter * restrict filter, uint64_t tag,
      uint64_t block_index) {
   uint64_t index = block_index / QUQU_BUCKETS_PER_BLOCK;
   uint64_t offset = block_index % QUQU_BUCKETS_PER_BLOCK;

#ifdef __AVX512BW__
#if TAG_BITS == 8
   __m512i bcast = _mm512_set1_epi8(tag);
   __m512i block =
      _mm512_loadu_si512(reinterpret_cast<__m512i*>(&filter->blocks[index].block));
   volatile __mmask64 result = _mm512_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);
#elif TAG_BITS == 16
   __m512i bcast = _mm512_set1_epi16(tag);
   __m512i block =
      _mm512_loadu_si512(reinterpret_cast<__m512i*>(&filter->blocks[index].block));
   volatile __mmask64 result = _mm512_cmp_epi16_mask(bcast, block, _MM_CMPINT_EQ);
#endif
#else
#if TAG_BITS == 8
   __m256i bcast = _mm256_set1_epi8(tag);
   __m256i block = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&filter->blocks[index].block));
   __m256i result1t = _mm256_cmpeq_epi8(bcast, block);
   __mmask32 result1 = _mm256_movemask_epi8(result1t);
   /*__mmask32 result1 = _mm256_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);*/
   block = _mm256_loadu_si256(reinterpret_cast<__m256i*>((uint8_t*)&filter->blocks[index].block+32));
   __m256i result2t = _mm256_cmpeq_epi8(bcast, block);
   __mmask32 result2 = _mm256_movemask_epi8(result2t);
   /*__mmask32 result2 = _mm256_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);*/
   uint64_t result = (uint64_t)result2 << 32 | (uint64_t)result1;
#elif TAG_BITS == 16
   uint64_t alt_mask = 0x55555555;
   __m256i bcast = _mm256_set1_epi16(tag);
   __m256i block = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&filter->blocks[index].block));
   __m256i result1t = _mm256_cmpeq_epi16(bcast, block);
   __mmask32 result1 = _mm256_movemask_epi8(result1t);
   result1 = _pext_u32(result1, alt_mask);
   /*__mmask32 result1 = _mm256_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);*/
   block = _mm256_loadu_si256(reinterpret_cast<__m256i*>((uint8_t*)&filter->blocks[index].block+32));
   __m256i result2t = _mm256_cmpeq_epi16(bcast, block);
   __mmask32 result2 = _mm256_movemask_epi8(result2t);
   result2 = _pext_u32(result2, alt_mask);
   /*__mmask32 result2 = _mm256_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);*/
   uint64_t result = (uint64_t)result2 << 16 | (uint64_t)result1;
#endif
#endif

   //printf("TAG %ld\n", tag);
   //print_block(filter, index);
   //printf("-----------------\n");

   if (result == 0) {
      // no matching tags, can bail
      return false;
   }

#if TAG_BITS == 8
   uint64_t start = offset != 0 ? lookup_128(filter->blocks[index].block.md, offset -
	 1) : one[0] << 2 * sizeof(uint64_t);
   uint64_t end = lookup_128(filter->blocks[index].block.md, offset);
#elif TAG_BITS == 16
   uint64_t start = offset != 0 ? lookup_64(filter->blocks[index].block.md, offset -
	 1) : one[0] << (sizeof(uint64_t)/2);
   uint64_t end = lookup_64(filter->blocks[index].block.md, offset);
#endif
   uint64_t mask = end - start;

   uint64_t check_indexes = mask & result;
   if (check_indexes != 0) { // remove the first available tag
      linked_blocks    * restrict blocks             = filter->blocks;

      // original
      /*
	 uint64_t remove_index = __builtin_ctzll(check_indexes);
	 remove_tags_512(&blocks[index], remove_index);
#if TAG_BITS == 8
remove_index = remove_index + offset - sizeof(__uint128_t);
uint64_t *block_md = blocks[block_index / QUQU_BUCKETS_PER_BLOCK].md;
remove_md(block_md, remove_index);
#elif TAG_BITS == 16
remove_index = remove_index + offset - sizeof(uint64_t);
uint64_t *block_md = &blocks[block_index / QUQU_BUCKETS_PER_BLOCK].md;
remove_md(block_md, remove_index);
#endif
return true;
*/

      // CVQF
      // check check_tags for comment

      uint64_t slot_start = _tzcnt_u64(start);
      uint64_t slot_end = _tzcnt_u64(end);
      uint64_t slot_check;
      uint64_t remove_index = __builtin_ctzll(check_indexes) + offset - sizeof(__uint128_t);
#if TAG_BITS == 8
      uint64_t *block_md = blocks[block_index / QUQU_BUCKETS_PER_BLOCK].block.md;
#elif TAG_BITS == 16
      uint64_t *block_md = &blocks[block_index / QUQU_BUCKETS_PER_BLOCK].block.md;
#endif

      if (tag == QUQU_MAX) {
        if (((check_indexes >> (slot_end - 1)) & 1) == 1) {
          slot_check = slot_end - 1;
	        // can only be only one 255
          if (slot_check == slot_start) {
            remove_tags_512(&blocks[index].block, slot_check);
            remove_md(block_md, remove_index);
            //print_block(filter, index);
            return true;
          }

          // check the one before slot_check
          else {
            // if it is 255
            if (blocks[index].block.tags[slot_check - 1 - QUQU_PRESLOT] == QUQU_MAX) {
              remove_tags_512(&blocks[index].block, slot_check);
              remove_md(block_md, remove_index);
              //print_block(filter, index);
              return true;
            }

            // if it is 0
            else if (blocks[index].block.tags[slot_check - 1 - QUQU_PRESLOT] == 0) {
              remove_tags_512(&blocks[index].block, slot_check);
              remove_md(block_md, remove_index);
              //print_block(filter, index);
              return true;
            }

            // if not
            else {
              // cannot make sequence
              if (slot_check == slot_start + 1) {
                remove_tags_512(&blocks[index].block, slot_check);
                remove_md(block_md, remove_index);
                //print_block(filter, index);
                return true;
              }
              // could be counter sequence embedded
              else {
                uint8_t temp_tag = blocks[index].block.tags[slot_check - 1 - QUQU_PRESLOT];
                uint64_t slot_temp = slot_check - 2;
                while (slot_temp != slot_start) {
                  if (blocks[index].block.tags[slot_temp - QUQU_PRESLOT] == QUQU_MAX) {
                  // 255 is not counter
                    if (blocks[index].block.tags[slot_temp - 1 - QUQU_PRESLOT] == QUQU_MAX) {
                      remove_tags_512(&blocks[index].block, slot_check);
                      remove_md(block_md, remove_index);
                      //print_block(filter, index);
                      return true;
                    }

                    // 255 might not be counter
                    else if (blocks[index].block.tags[slot_temp - 1 - QUQU_PRESLOT] == 0) {
                      // only one 0
                      if (slot_temp == slot_start + 1) {
                        temp_tag--;
                        // need to remove counter
                        if (temp_tag == 0) {
                          remove_tags_512(&blocks[index].block, slot_check - 1);
                          remove_md(block_md, remove_index);
                          //print_block(filter, index);
                          return true;
                        }

                        // decrease 1 counter
                        else {
                          blocks[index].block.tags[slot_check - 1 - QUQU_PRESLOT] = temp_tag;
                          //print_block(filter, index);
                          return true;
                        }
                      }

                      // something even before
                      else {
                        // 255 is not counter
                        if (blocks[index].block.tags[slot_temp - 2 - QUQU_PRESLOT] == temp_tag) {
                          remove_tags_512(&blocks[index].block, slot_check);
                          remove_md(block_md, remove_index);
                          //print_block(filter, index);
                          return true;
                        }
                        // decrease counter
                        else {
                          temp_tag--;
                          // need to remove counter
                          if (temp_tag == 0) {
                            remove_tags_512(&blocks[index].block, slot_check - 1);
                            remove_md(block_md, remove_index);
                            //print_block(filter, index);
                            return true;
                          }
                          // decrease 1 counter
                          else {
                            blocks[index].block.tags[slot_check - 1 - QUQU_PRESLOT] = temp_tag;
                            //print_block(filter, index);
                            return true;
                          }
                        }
                      }
                    }
                    // counter embedded for sure
                    else {
                      temp_tag--;
                        // need to remove 1 counter
                        if (temp_tag == 0) {
                          remove_tags_512(&blocks[index].block, slot_check - 1);
                          remove_md(block_md, remove_index);
                          //print_block(filter, index);
                          return true;
                        }
                        // decrease 1 counter
                        else {
                          blocks[index].block.tags[slot_check - 1 - QUQU_PRESLOT] = temp_tag;
                          //print_block(filter, index);
                          return true;
                        }
                      }
                    }
                    // not a counter sequence
                    if (blocks[index].block.tags[slot_temp - QUQU_PRESLOT] < QUQU_MAX - 1) {
                      remove_tags_512(&blocks[index].block, slot_check);
                      remove_md(block_md, remove_index);
                      //print_block(filter, index);
                      return true;
                    }
                    slot_temp--;
                  } // end of while

                // a big counter, consuming all bucket space
                if (blocks[index].block.tags[slot_temp - QUQU_PRESLOT] == QUQU_MAX) {
                  temp_tag--;
                  // need to remove 1 counter
                  if (temp_tag == 0) {
                    remove_tags_512(&blocks[index].block, slot_check - 1);
                    remove_md(block_md, remove_index);
                    //print_block(filter, index);
                    return true;
                  }
                  // decrease 1 counter
                  else {
                    blocks[index].block.tags[slot_check - 1 - QUQU_PRESLOT] = temp_tag;
                    //print_block(filter, index);
                    return true;
                  }
                }
                // while-d till end
                else {
                  remove_tags_512(&blocks[index].block, slot_check);
                  remove_md(block_md, remove_index);
                  //print_block(filter, index);
                  return true;
                }
              }
            }
          }
        }
        else {
          return false;
        }
      }
      else if (tag == 0) {
        if (((check_indexes >> slot_start) & 1) == 1) {
          slot_check = slot_start;
          // only one item, 0
          if (slot_end == slot_start + 1) {
            remove_tags_512(&blocks[index].block, slot_check);
            remove_md(block_md, remove_index);
            //print_block(filter, index);
            return true;
          }
          // more than one item
          else {
            uint8_t temp_tag = blocks[index].block.tags[slot_check + 1 - QUQU_PRESLOT];
            uint64_t slot_temp = slot_check + 1;
            // if 0, 0, ...
            if (temp_tag == 0) {
              remove_tags_512(&blocks[index].block, slot_check);
              remove_md(block_md, remove_index);
              //print_block(filter, index);
              return true;
            }
            // if 0, value, ...
            else {
              // cannot make sequence
              if (slot_end < slot_start + 4) {
                remove_tags_512(&blocks[index].block, slot_check);
                remove_md(block_md, remove_index);
                //print_block(filter, index);
                return true;
              }
              // maybe long counter
              else if (temp_tag == QUQU_MAX) {
                while (blocks[index].block.tags[slot_temp - QUQU_PRESLOT] == QUQU_MAX) {
                  slot_temp++;
                  if (slot_temp == slot_end) {
                    remove_tags_512(&blocks[index].block, slot_check);
                    remove_md(block_md, remove_index);
                    //print_block(filter, index);
                    return true;
                  }
                }
                temp_tag = blocks[index].block.tags[slot_temp - QUQU_PRESLOT];
                if (temp_tag == 0) {
                  if (slot_temp > slot_end - 2) {
                    remove_tags_512(&blocks[index].block, slot_check);
                    remove_md(block_md, remove_index);
                    //print_block(filter, index);
                    return true;
                  }
                  else {
                    if (blocks[index].block.tags[slot_temp + 1 - QUQU_PRESLOT] == 0) {
                      blocks[index].block.tags[slot_temp - 1 - QUQU_PRESLOT]--;
                      //print_block(filter, index);
                      return true;
                    }
                    else {
                      remove_tags_512(&blocks[index].block, slot_check);
                      remove_md(block_md, remove_index);
                      //print_block(filter, index);
                      return true;
                    }
                  }
                }
                else {
                  if (slot_temp > slot_end - 3) {
                    remove_tags_512(&blocks[index].block, slot_check);
                    remove_md(block_md, remove_index);
                    //print_block(filter, index);
                    return true;
                  }
                  else {
                    if (  (blocks[index].block.tags[slot_temp + 1 - QUQU_PRESLOT] == 0)
                      && (blocks[index].block.tags[slot_temp + 2 - QUQU_PRESLOT] == 0)) {
                      temp_tag--;
                      if (temp_tag == 0) {
                        remove_tags_512(&blocks[index].block, slot_temp);
                        remove_md(block_md, remove_index);
                        //print_block(filter, index);
                        return true;
                      }
                      else {
                        blocks[index].block.tags[slot_temp - QUQU_PRESLOT] = temp_tag;
                        //print_block(filter, index);
                        return true;
                      }
                    }
                    else {
                      remove_tags_512(&blocks[index].block, slot_check);
                      remove_md(block_md, remove_index);
                      //print_block(filter, index);
                      return true;
                    }
                  }
                }
              }

              // could be short counter
              else {
                // counter
                if (  (blocks[index].block.tags[slot_temp + 1 - QUQU_PRESLOT] == 0)
                  && (blocks[index].block.tags[slot_temp + 2 - QUQU_PRESLOT] == 0)) {
                  temp_tag--;
                  if (temp_tag == 0) {
                    remove_tags_512(&blocks[index].block, slot_temp);
                    remove_md(block_md, remove_index);
                    //print_block(filter, index);
                    return true;
                  }
                  else {
                    blocks[index].block.tags[slot_temp - QUQU_PRESLOT] = temp_tag;
                    //print_block(filter, index);
                    return true;
                  }
                }
                // not counter
                else {
                  remove_tags_512(&blocks[index].block, slot_check);
                  remove_md(block_md, remove_index);
                  //print_block(filter, index);
                  return true;
                }
              }
            }
          }
        }
        else {
          return false;
        }
      }
      else {
        while (check_indexes != 0) {
          slot_check = _tzcnt_u64(check_indexes);
          if (slot_check == slot_start) {
            //ultimatum
            if (slot_check == slot_end - 1) {
              remove_tags_512(&blocks[index].block, slot_check);
              remove_md(block_md, remove_index);
              //print_block(filter, index);
              return true;
            }
            else {
              uint64_t slot_temp = slot_check + 1;
              uint8_t temp_tag = blocks[index].block.tags[slot_temp - QUQU_PRESLOT];
              if (temp_tag == 0) {
                check_indexes &= ~(one[0] << slot_check);
                slot_temp = _tzcnt_u64(check_indexes);
                if (slot_temp >= slot_end) {
                  printf("ERROR1\n");
                  return false;
                }
                else {
                  slot_temp--;
                  temp_tag = blocks[index].block.tags[slot_temp - QUQU_PRESLOT];
                  temp_tag--;
                  if (temp_tag == tag) {
                    temp_tag--;
                  if (temp_tag == 0) {
                    if (slot_temp == slot_check + 2) {
                      remove_tags_512(&blocks[index].block, slot_check);
                      remove_md(block_md, remove_index);
                      remove_tags_512(&blocks[index].block, slot_check);
                      remove_md(block_md, remove_index);
                      //print_block(filter, index);
                      return true;
                    }
                    else {
                      remove_tags_512(&blocks[index].block, slot_temp);
                      remove_md(block_md, remove_index);
                      //print_block(filter, index);
                      return true;
                    }
                  }
                  else {
                    blocks[index].block.tags[slot_temp - QUQU_PRESLOT] = temp_tag;
                    if (slot_temp == slot_check + 2) {
                      remove_tags_512(&blocks[index].block, slot_check + 1);
                      remove_md(block_md, remove_index);
                    }
                    //print_block(filter, index);
                    return true;
                  }
                }
                else if (temp_tag == 0) {
                  remove_tags_512(&blocks[index].block, slot_temp);
                  remove_md(block_md, remove_index);
                  //print_block(filter, index);
                  return true;
                }
                else {
                  blocks[index].block.tags[slot_temp - QUQU_PRESLOT] = temp_tag;
                  //print_block(filter, index);
                  return true;
                }
              }
            }
            else if (temp_tag < tag) {
              temp_tag--;
              if (temp_tag == 0) {
                remove_tags_512(&blocks[index].block, slot_temp);
                remove_md(block_md, remove_index);
                //print_block(filter, index);
                return true;
              }
              else {
                blocks[index].block.tags[slot_temp - QUQU_PRESLOT] = temp_tag;
                //print_block(filter, index);
                return true;
              }
            }
            // greater than or equal to
            else {
              remove_tags_512(&blocks[index].block, slot_check);
              remove_md(block_md, remove_index);
              //print_block(filter, index);
              return true;
            }
          }
          // ultimatum
        }
        else if (slot_check == slot_end - 1) {
          remove_tags_512(&blocks[index].block, slot_check);
          remove_md(block_md, remove_index);
          //print_block(filter, index);
          return true;
        }
        else {
          if (blocks[index].block.tags[slot_check - 1 - QUQU_PRESLOT] > tag) {
          }
          else if (blocks[index].block.tags[slot_check - 1 - QUQU_PRESLOT] == 0) {
            if (slot_check == slot_start + 1) {
              if (slot_check < slot_end - 2) {
                if (blocks[index].block.tags[slot_check + 1 - QUQU_PRESLOT] == 0
                  && blocks[index].block.tags[slot_check + 2 - QUQU_PRESLOT] == 0) {
                }
                else {
                // NO THINK
                //ultimatum
                  if (slot_check == slot_end - 1) {
                    remove_tags_512(&blocks[index].block, slot_check);
                    remove_md(block_md, remove_index);
                    //print_block(filter, index);
                    return true;
                  }
                  else {
                    uint64_t slot_temp = slot_check + 1;
                    uint8_t temp_tag = blocks[index].block.tags[slot_temp - QUQU_PRESLOT];
                    if (temp_tag == 0) {
                      check_indexes &= ~(one[0] << slot_check);
                      slot_temp = _tzcnt_u64(check_indexes);
                      if (slot_temp >= slot_end) {
                        printf("ERROR2\n");
                        return false;
                      }
                      else {
                        slot_temp--;
                        temp_tag = blocks[index].block.tags[slot_temp - QUQU_PRESLOT];
                        temp_tag--;
                        if (temp_tag == tag) {
                          temp_tag--;
                          if (temp_tag == 0) {
                            if (slot_temp == slot_check + 2) {
                              remove_tags_512(&blocks[index].block, slot_check);
                              remove_md(block_md, remove_index);
                              remove_tags_512(&blocks[index].block, slot_check);
                              remove_md(block_md, remove_index);
                              //print_block(filter, index);
                              return true;
                            }
                            else {
                              remove_tags_512(&blocks[index].block, slot_temp);
                              remove_md(block_md, remove_index);
                              //print_block(filter, index);
                              return true;
                            }
                          }
                          else {
                            blocks[index].block.tags[slot_temp - QUQU_PRESLOT] = temp_tag;
                            if (slot_temp == slot_check + 2) {
                              remove_tags_512(&blocks[index].block, slot_check + 1);
                              remove_md(block_md, remove_index);
                            }
                            //print_block(filter, index);
                            return true;
                          }
                        }
                        else if (temp_tag == 0) {
                          remove_tags_512(&blocks[index].block, slot_temp);
                          remove_md(block_md, remove_index);
                          //print_block(filter, index);
                          return true;
                        }
                        else {
                          blocks[index].block.tags[slot_temp - QUQU_PRESLOT] = temp_tag;
                          //print_block(filter, index);
                          return true;
                        }
                      }
                    }
                    else if (temp_tag < tag) {
                      temp_tag--;
                      if (temp_tag == 0) {
                        remove_tags_512(&blocks[index].block, slot_temp);
                        remove_md(block_md, remove_index);
                        //print_block(filter, index);
                        return true;
                       }
                       else {
                         blocks[index].block.tags[slot_temp - QUQU_PRESLOT] = temp_tag;
                         //print_block(filter, index);
                         return true;
                       }
                     }
                     // greater than or equal to
                     else {
                       remove_tags_512(&blocks[index].block, slot_check);
                       remove_md(block_md, remove_index);
                       //print_block(filter, index);
                       return true;
                     }
                   }
                   // ultimatum
                   // NO THINK
                 }
               }
               else {
                 // NO THINK
                 //ultimatum
                 if (slot_check == slot_end - 1) {
                   remove_tags_512(&blocks[index].block, slot_check);
                   remove_md(block_md, remove_index);
                   //print_block(filter, index);
                   return true;
                 }
                 else {
                   uint64_t slot_temp = slot_check + 1;
                   uint8_t temp_tag = blocks[index].block.tags[slot_temp - QUQU_PRESLOT];
                   if (temp_tag == 0) {
                     check_indexes &= ~(one[0] << slot_check);
                     slot_temp = _tzcnt_u64(check_indexes);
                     if (slot_temp >= slot_end) {
                       printf("ERROR3\n");
                       return false;
                     }
                     else {
                       slot_temp--;
                       temp_tag = blocks[index].block.tags[slot_temp - QUQU_PRESLOT];
                       temp_tag--;
                       if (temp_tag == tag) {
                         temp_tag--;
                         if (temp_tag == 0) {
                           if (slot_temp == slot_check + 2) {
                             remove_tags_512(&blocks[index].block, slot_check);
                             remove_md(block_md, remove_index);
                             remove_tags_512(&blocks[index].block, slot_check);
                             remove_md(block_md, remove_index);
                             //print_block(filter, index);
                             return true;
                           }
                           else {
                             remove_tags_512(&blocks[index].block, slot_temp);
                             remove_md(block_md, remove_index);
                             //print_block(filter, index);
                             return true;
                           }
                         }
                         else {
                           blocks[index].block.tags[slot_temp - QUQU_PRESLOT] = temp_tag;
                           if (slot_temp == slot_check + 2) {
                             remove_tags_512(&blocks[index].block, slot_check + 1);
                             remove_md(block_md, remove_index);
                           }
                           //print_block(filter, index);
                           return true;
                         }
                       }
                       else if (temp_tag == 0) {
                         remove_tags_512(&blocks[index].block, slot_temp);
                         remove_md(block_md, remove_index);
                         //print_block(filter, index);
                         return true;
                       }
				 else {
				    blocks[index].block.tags[slot_temp - QUQU_PRESLOT] = temp_tag;
				    //print_block(filter, index);
				    return true;
				 }
			      }
			   }
			   else if (temp_tag < tag) {
			      temp_tag--;
			      if (temp_tag == 0) {
				 remove_tags_512(&blocks[index].block, slot_temp);
				 remove_md(block_md, remove_index);
				 //print_block(filter, index);
				 return true;
			      }
			      else {
				 blocks[index].block.tags[slot_temp - QUQU_PRESLOT] = temp_tag;
				 //print_block(filter, index);
				 return true;
			      }
			   }
			   // greater than or equal to
			   else {
			      remove_tags_512(&blocks[index].block, slot_check);
			      remove_md(block_md, remove_index);
			      //print_block(filter, index);
			      return true;
			   }
			}
			// ultimatum
			// NO THINK
		     }
		  }
		  else {
		     if (blocks[index].block.tags[slot_check - 2 - QUQU_PRESLOT] == 0) {
			// NO THINK
			//ultimatum
			if (slot_check == slot_end - 1) {
			   remove_tags_512(&blocks[index].block, slot_check);
			   remove_md(block_md, remove_index);
			   //print_block(filter, index);
			   return true;
			}
			else {
			   uint64_t slot_temp = slot_check + 1;
			   uint8_t temp_tag = blocks[index].block.tags[slot_temp - QUQU_PRESLOT];
			   if (temp_tag == 0) {
			      check_indexes &= ~(one[0] << slot_check);
			      slot_temp = _tzcnt_u64(check_indexes);
			      if (slot_temp >= slot_end) {
				 printf("ERROR4\n");
				 return false;
			      }
			      else {
				 slot_temp--;
				 temp_tag = blocks[index].block.tags[slot_temp - QUQU_PRESLOT];
				 temp_tag--;
				 if (temp_tag == tag) {
				    temp_tag--;
				    if (temp_tag == 0) {
				       if (slot_temp == slot_check + 2) {
					  remove_tags_512(&blocks[index].block, slot_check);
					  remove_md(block_md, remove_index);
					  remove_tags_512(&blocks[index].block, slot_check);
					  remove_md(block_md, remove_index);
					  //print_block(filter, index);
					  return true;
				       }
				       else {
					  remove_tags_512(&blocks[index].block, slot_temp);
					  remove_md(block_md, remove_index);
					  //print_block(filter, index);
					  return true;
				       }
				    }
				    else {
				       blocks[index].block.tags[slot_temp - QUQU_PRESLOT] = temp_tag;
				       if (slot_temp == slot_check + 2) {
					  remove_tags_512(&blocks[index].block, slot_check + 1);
					  remove_md(block_md, remove_index);
				       }
				       //print_block(filter, index);
				       return true;
				    }
				 }
				 else if (temp_tag == 0) {
				    remove_tags_512(&blocks[index].block, slot_temp);
				    remove_md(block_md, remove_index);
				    //print_block(filter, index);
				    return true;
				 }
				 else {
				    blocks[index].block.tags[slot_temp - QUQU_PRESLOT] = temp_tag;
				    //print_block(filter, index);
				    return true;
				 }
			      }
			   }
			   else if (temp_tag < tag) {
			      temp_tag--;
			      if (temp_tag == 0) {
				 remove_tags_512(&blocks[index].block, slot_temp);
				 remove_md(block_md, remove_index);
				 //print_block(filter, index);
				 return true;
			      }
			      else {
				 blocks[index].block.tags[slot_temp - QUQU_PRESLOT] = temp_tag;
				 //print_block(filter, index);
				 return true;
			      }
			   }
			   // greater than or equal to
			   else {
			      remove_tags_512(&blocks[index].block, slot_check);
			      remove_md(block_md, remove_index);
			      //print_block(filter, index);
			      return true;
			   }
			}
			// ultimatum
			// NO THINK
		     }
		     else {
		     }
		  }
	       }
	       else if (blocks[index].block.tags[slot_check - 1 - QUQU_PRESLOT] < tag) {
		  // NO THINK
		  //ultimatum
		  if (slot_check == slot_end - 1) {
		     remove_tags_512(&blocks[index].block, slot_check);
		     remove_md(block_md, remove_index);
		     //print_block(filter, index);
		     return true;
		  }
		  else {
		     uint64_t slot_temp = slot_check + 1;
		     uint8_t temp_tag = blocks[index].block.tags[slot_temp - QUQU_PRESLOT];
		     if (temp_tag == 0) {
			check_indexes &= ~(one[0] << slot_check);
			slot_temp = _tzcnt_u64(check_indexes);
			if (slot_temp >= slot_end) {
			   printf("ERROR5\n");
			   return false;
			}
			else {
			   slot_temp--;
			   temp_tag = blocks[index].block.tags[slot_temp - QUQU_PRESLOT];
			   temp_tag--;
			   if (temp_tag == tag) {
			      temp_tag--;
			      if (temp_tag == 0) {
				 if (slot_temp == slot_check + 2) {
				    remove_tags_512(&blocks[index].block, slot_check);
				    remove_md(block_md, remove_index);
				    remove_tags_512(&blocks[index].block, slot_check);
				    remove_md(block_md, remove_index);
				    //print_block(filter, index);
				    return true;
				 }
				 else {
				    remove_tags_512(&blocks[index].block, slot_temp);
				    remove_md(block_md, remove_index);
				    //print_block(filter, index);
				    return true;
				 }
			      }
			      else {
				 blocks[index].block.tags[slot_temp - QUQU_PRESLOT] = temp_tag;
				 if (slot_temp == slot_check + 2) {
				    remove_tags_512(&blocks[index].block, slot_check + 1);
				    remove_md(block_md, remove_index);
				 }
				 //print_block(filter, index);
				 return true;
			      }
			   }
			   else if (temp_tag == 0) {
			      remove_tags_512(&blocks[index].block, slot_temp);
			      remove_md(block_md, remove_index);
			      //print_block(filter, index);
			      return true;
			   }
			   else {
			      blocks[index].block.tags[slot_temp - QUQU_PRESLOT] = temp_tag;
			      //print_block(filter, index);
			      return true;
			   }
			}
		     }
		     else if (temp_tag < tag) {
			temp_tag--;
			if (temp_tag == 0) {
			   remove_tags_512(&blocks[index].block, slot_temp);
			   remove_md(block_md, remove_index);
			   //print_block(filter, index);
			   return true;
			}
			else {
			   blocks[index].block.tags[slot_temp - QUQU_PRESLOT] = temp_tag;
			   //print_block(filter, index);
			   return true;
			}
		     }
		     // greater than or equal to
		     else {
			remove_tags_512(&blocks[index].block, slot_check);
			remove_md(block_md, remove_index);
			//print_block(filter, index);
			return true;
		     }
		  }
		  // ultimatum
		  // NO THINK
	       }
	       else {
	       }
	    }
	    check_indexes &= ~(one[0] << slot_check);
	 }
      }
      //return true;

      // if every matching tags are counters
      return false;
   } else
      return false;
}

bool vqf_remove(vqf_filter * restrict filter, uint64_t hash) {
   vqf_metadata * restrict metadata           = &filter->metadata;
   uint64_t                 key_remainder_bits = metadata->key_remainder_bits;
   uint64_t                 range              = metadata->range;

   uint64_t block_index = hash >> key_remainder_bits;
   uint64_t tag = hash & TAG_MASK;
   uint64_t alt_block_index = ((hash ^ (tag * 0x5bd1e995)) % range) >> key_remainder_bits;

   __builtin_prefetch(&filter->blocks[alt_block_index / QUQU_BUCKETS_PER_BLOCK].block);

   return remove_tags(filter, tag, block_index) || remove_tags(filter, tag, alt_block_index);
}

static inline bool check_tags(vqf_filter * restrict filter, uint64_t tag,
      uint64_t block_index) {
   uint64_t index = block_index / QUQU_BUCKETS_PER_BLOCK;
   uint64_t offset = block_index % QUQU_BUCKETS_PER_BLOCK;

#ifdef __AVX512BW__
#if TAG_BITS == 8
   __m512i bcast = _mm512_set1_epi8(tag);
   __m512i block =
      _mm512_loadu_si512(reinterpret_cast<__m512i*>(&filter->blocks[index]));
   volatile __mmask64 result = _mm512_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);
#elif TAG_BITS == 16
   __m512i bcast = _mm512_set1_epi16(tag);
   __m512i block =
      _mm512_loadu_si512(reinterpret_cast<__m512i*>(&filter->blocks[index]));
   volatile __mmask64 result = _mm512_cmp_epi16_mask(bcast, block, _MM_CMPINT_EQ);
#endif
#else
#if TAG_BITS == 8
   __m256i bcast = _mm256_set1_epi8(tag);
   __m256i block = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&filter->blocks[index]));
   __m256i result1t = _mm256_cmpeq_epi8(bcast, block);
   __mmask32 result1 = _mm256_movemask_epi8(result1t);
   /*__mmask32 result1 = _mm256_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);*/
   block = _mm256_loadu_si256(reinterpret_cast<__m256i*>((uint8_t*)&filter->blocks[index]+32));
   __m256i result2t = _mm256_cmpeq_epi8(bcast, block);
   __mmask32 result2 = _mm256_movemask_epi8(result2t);
   /*__mmask32 result2 = _mm256_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);*/
   uint64_t result = (uint64_t)result2 << 32 | (uint64_t)result1;
#elif TAG_BITS == 16
   uint64_t alt_mask = 0x55555555;
   __m256i bcast = _mm256_set1_epi16(tag);
   __m256i block = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&filter->blocks[index]));
   __m256i result1t = _mm256_cmpeq_epi16(bcast, block);
   __mmask32 result1 = _mm256_movemask_epi8(result1t);
   result1 = _pext_u32(result1, alt_mask);
   /*__mmask32 result1 = _mm256_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);*/
   block = _mm256_loadu_si256(reinterpret_cast<__m256i*>((uint8_t*)&filter->blocks[index]+32));
   __m256i result2t = _mm256_cmpeq_epi16(bcast, block);
   __mmask32 result2 = _mm256_movemask_epi8(result2t);
   result2 = _pext_u32(result2, alt_mask);
   /*__mmask32 result2 = _mm256_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);*/
   uint64_t result = (uint64_t)result2 << 16 | (uint64_t)result1;
#endif
#endif

   if (result == 0) {
      // no matching tags, can bail
      return false;
   }

#if TAG_BITS == 8
   uint64_t start = offset != 0 ? lookup_128(filter->blocks[index].block.md, offset -
         1) : one[0] << 2 * sizeof(uint64_t); // 1 << 16
   uint64_t end = lookup_128(filter->blocks[index].block.md, offset);
#elif TAG_BITS == 16
   uint64_t start = offset != 0 ? lookup_64(filter->blocks[index].block.md, offset -
	 1) : one[0] << (sizeof(uint64_t)/2); // 1 << 4
   uint64_t end = lookup_64(filter->blocks[index].md, offset);
#endif
   uint64_t mask = end - start;

   //return (mask & result) != 0; //turn off for CVQF

   // equal locations, tag order [tag64 tag63 ... tag16]
   // tag value is tag
   // [slot_start, slot_end)
   linked_blocks * restrict blocks = filter->blocks;
   uint64_t equalLocations = mask & result;
   uint64_t slot_start = _tzcnt_u64(start);
   uint64_t slot_end = _tzcnt_u64(end) - 1;
   uint64_t slot_check;

   // 255 should be last tag
   if (tag == QUQU_MAX) {
      if (((equalLocations >> slot_end) & 1 ) == 1)
	 return true;
      else
	 return false;
   }

   // 0 should be first tag
   else if (tag == 0) {
      if (((equalLocations >> slot_start) & 1 ) == 1)
	 return true;
      else
	 return false;
   }

   // other tags
   else {
      // filter->blocks[index].tags[slot_check - 16];
      while (equalLocations != 0) {
	 // only check necessaries
	 slot_check = _tzcnt_u64(equalLocations);
	 // if first
	 if (slot_check == slot_start)
	    return true;
	 // if last
	 else if (slot_check == slot_end)
	    return true;
	 // not first, nor last
	 else {
	    // the escape sequence
	    if (blocks[index].block.tags[slot_check - 1 - QUQU_PRESLOT] > tag) {
	       // counter
	    }
	    // [... 0, tag ...]
	    else if (blocks[index].block.tags[slot_check - 1 - QUQU_PRESLOT] == 0) {
	       // [0, tag ...]
	       if (slot_check == slot_start + 1) {
		  if (slot_check < slot_end - 1) {
		     // [0, tag, 0, 0 ...]
		     if (blocks[index].block.tags[slot_check + 1 - QUQU_PRESLOT] == 0
			   && blocks[index].block.tags[slot_check + 2 - QUQU_PRESLOT] == 0) {
			// counter of '0'
		     }
		     // not [0, tag, 0, 0 ...] sequence
		     else
			return true;
		  }
		  // current bucket has only 3 slots. the first slot is 0. cannot make a counter slot with two slots. this slot is not a counter.
		  else
		     return true;
	       }
	       // [... 0, tag ...]
	       else {
		  // [ ... 0, 0, tag ...]
		  if (blocks[index].block.tags[slot_check - 2 - QUQU_PRESLOT] == 0)
		     return true;
		  else {
		     // counter
		  }
	       }
	    }
	    // tag before is less than
	    else if (blocks[index].block.tags[slot_check - 1 - QUQU_PRESLOT] < tag)
	       return true;
	    // tag before is equal to
	    else {
/*              printf("[CYDBG] tag: %ld, offset: %ld\n", tag, offset);
              print_block(filter, index);*/
	    }
	 }
	 equalLocations &= ~(one[0] << slot_check);
      }
   }
   return false;
}

int count_tags(vqf_filter * restrict filter, uint64_t tag, uint64_t block_index){
   uint64_t index = block_index / QUQU_BUCKETS_PER_BLOCK;
   uint64_t offset = block_index % QUQU_BUCKETS_PER_BLOCK;

#ifdef __AVX512BW__
#if TAG_BITS == 8
   __m512i bcast = _mm512_set1_epi8(tag);
   __m512i block =
      _mm512_loadu_si512(reinterpret_cast<__m512i*>(&filter->blocks[index].block));
   volatile __mmask64 result = _mm512_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);
#elif TAG_BITS == 16
   __m512i bcast = _mm512_set1_epi16(tag);
   __m512i block =
      _mm512_loadu_si512(reinterpret_cast<__m512i*>(&filter->blocks[index].block));
   volatile __mmask64 result = _mm512_cmp_epi16_mask(bcast, block, _MM_CMPINT_EQ);
#endif
#else
#if TAG_BITS == 8
   __m256i bcast = _mm256_set1_epi8(tag);
   __m256i block = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&filter->blocks[index].block));
   __m256i result1t = _mm256_cmpeq_epi8(bcast, block);
   __mmask32 result1 = _mm256_movemask_epi8(result1t);
   /*__mmask32 result1 = _mm256_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);*/
   block = _mm256_loadu_si256(reinterpret_cast<__m256i*>((uint8_t*)&filter->blocks[index].block+32));
   __m256i result2t = _mm256_cmpeq_epi8(bcast, block);
   __mmask32 result2 = _mm256_movemask_epi8(result2t);
   /*__mmask32 result2 = _mm256_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);*/
   uint64_t result = (uint64_t)result2 << 32 | (uint64_t)result1;
#elif TAG_BITS == 16
   uint64_t alt_mask = 0x55555555;
   __m256i bcast = _mm256_set1_epi16(tag);
   __m256i block = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&filter->blocks[index].block));
   __m256i result1t = _mm256_cmpeq_epi16(bcast, block);
   __mmask32 result1 = _mm256_movemask_epi8(result1t);
   result1 = _pext_u32(result1, alt_mask);
   /*__mmask32 result1 = _mm256_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);*/
   block = _mm256_loadu_si256(reinterpret_cast<__m256i*>((uint8_t*)&filter->blocks[index].block+32));
   __m256i result2t = _mm256_cmpeq_epi16(bcast, block);
   __mmask32 result2 = _mm256_movemask_epi8(result2t);
   result2 = _pext_u32(result2, alt_mask);
   /*__mmask32 result2 = _mm256_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);*/
   uint64_t result = (uint64_t)result2 << 16 | (uint64_t)result1;
#endif
#endif

   //printf("TAG %ld\n", tag);
   //print_block(filter, index);
   //printf("-----------------\n");

   if (result == 0) {
      // no matching tags, can bail
      return 0;
   }

#if TAG_BITS == 8
   uint64_t start = offset != 0 ? lookup_128(filter->blocks[index].block.md, offset -
	 1) : one[0] << 2 * sizeof(uint64_t);
   uint64_t end = lookup_128(filter->blocks[index].block.md, offset);
#elif TAG_BITS == 16
   uint64_t start = offset != 0 ? lookup_64(filter->blocks[index].block.md, offset -
	 1) : one[0] << (sizeof(uint64_t)/2);
   uint64_t end = lookup_64(filter->blocks[index].block.md, offset);
#endif
   uint64_t mask = end - start;

   uint64_t check_indexes = mask & result;

   if (word_rank(check_indexes) == 1) /*CY. only 1 tag*/
      return 1;

   if (check_indexes != 0) { // remove the first available tag
      linked_blocks    * restrict blocks             = filter->blocks;

      // CVQF
      // check check_tags for comment

      uint64_t slot_start = _tzcnt_u64(start);
      uint64_t slot_end = _tzcnt_u64(end);
      uint64_t slot_check;
      uint64_t remove_index = __builtin_ctzll(check_indexes) + offset - sizeof(__uint128_t);
#if TAG_BITS == 8
      uint64_t *block_md = blocks[block_index / QUQU_BUCKETS_PER_BLOCK].block.md;
#elif TAG_BITS == 16
      uint64_t *block_md = &blocks[block_index / QUQU_BUCKETS_PER_BLOCK].block.md;
#endif

      if (tag == QUQU_MAX) {
        if (((check_indexes >> (slot_end - 1)) & 1) == 1) {
          slot_check = slot_end - 1;
          // can only be only one 255
          if (slot_check == slot_start) {
            return 1;
          }
          // check the one before slot_check
          else {
          // if it is 255
            if (blocks[index].block.tags[slot_check - 1 - QUQU_PRESLOT] == QUQU_MAX) {
              return 2;
            }
            // if it is 0
            else if (blocks[index].block.tags[slot_check - 1 - QUQU_PRESLOT] == 0) {
              return 1;
            }
            // if not
            else {
            // cannot make sequence
              if (slot_check == slot_start + 1) {
                return 1;
              }
              // could be counter sequence embedded
              else {
                uint8_t temp_tag = blocks[index].block.tags[slot_check - 1 - QUQU_PRESLOT];
                uint64_t slot_temp = slot_check - 2;
                while (slot_temp != slot_start) {
                  if (blocks[index].block.tags[slot_temp - QUQU_PRESLOT] == QUQU_MAX) {
                  // 255 is not counter
                    if (blocks[index].block.tags[slot_temp - 1 - QUQU_PRESLOT] == QUQU_MAX) {
                      return 1;
                    }
                    // 255 might not be counter
                    else if (blocks[index].block.tags[slot_temp - 1 - QUQU_PRESLOT] == 0) {
                      // only one 0
                      if (slot_temp == slot_start + 1) {
                        int sum = 0;
                        for (uint64_t i = slot_temp + 1; i < slot_check; i++) {
                          sum += blocks[index].block.tags[i - QUQU_PRESLOT];
                        }
                        return 2 + sum;
                      }
                      // something even before
                      else {
                        // 255 is not counter
                        if (blocks[index].block.tags[slot_temp - 2 - QUQU_PRESLOT] == temp_tag) {
                          return 1;
                        }
                        // decrease counter
                        else {
                          int sum = 0;
                          for (uint64_t i = slot_temp + 1; i < slot_check; i++) {
                            sum += blocks[index].block.tags[i - QUQU_PRESLOT];
                          }
                          return 2 + sum;
                        }
                      }
                    }
                    // counter embedded for sure
                    else {
                      int sum = 0;
                      for (uint64_t i = slot_temp + 1; i < slot_check; i++) {
                        sum += blocks[index].block.tags[i - QUQU_PRESLOT];
                      }
                      return 2 + sum;
                    }
                  }
                  // not a counter sequence
                  if (blocks[index].block.tags[slot_temp - QUQU_PRESLOT] < QUQU_MAX - 1) {
                    return 1;
                  }
                  slot_temp--;
                } // end of while
                // a big counter, consuming all bucket space
                if (blocks[index].block.tags[slot_temp - QUQU_PRESLOT] == QUQU_MAX) {
                  int sum = 0;
                  for (uint64_t i = slot_temp + 1; i < slot_check; i++) {
                    sum += blocks[index].block.tags[i - QUQU_PRESLOT];
                  }
                  return 2 + sum;
                }
                // while-d till end
                else {
                  return 1;
                }
              }
            }
          }
        } else {
          return false;
        }
      }
      else if (tag == 0) {
        if (((check_indexes >> slot_start) & 1) == 1) {
          slot_check = slot_start;
          // only one item, 0
        if (slot_end == slot_start + 1) {
          return 1;
        }
        // more than one item
        else {
          uint8_t temp_tag = blocks[index].block.tags[slot_check + 1 - QUQU_PRESLOT];
          uint64_t slot_temp = slot_check + 1;
	        // if 0, 0, ...
          if (temp_tag == 0) {
            if (slot_temp + 1 == slot_end)
              return 2;
            else {
              if (blocks[index].block.tags[slot_temp + 1 - QUQU_PRESLOT] == 0)
                return 3;
              else
                return 2;
            }
         }
	       // if 0, value, ...
         else {
           // cannot make sequence
           if (slot_end < slot_start + 4) {
             return 1;
           }
           // maybe long counter
           else if (temp_tag == QUQU_MAX) {
             while (blocks[index].block.tags[slot_temp - QUQU_PRESLOT] == QUQU_MAX) {
               slot_temp++;
               if (slot_temp == slot_end) {
                 return 1;
               }
             }
             temp_tag = blocks[index].block.tags[slot_temp - QUQU_PRESLOT];
             if (temp_tag == 0) {
               if (slot_temp > slot_end - 2) {
                 return 1;
               }
               else {
                 if (blocks[index].block.tags[slot_temp + 1 - QUQU_PRESLOT] == 0) {
                   int sum = 0;
                   for (uint64_t i = slot_check + 1; i < slot_temp; i++) {
                     sum += blocks[index].block.tags[i - QUQU_PRESLOT];
                   }
                   return 3 + sum;
                 }
                 else {
                   return 1;
                 }
               }
             }
             else {
               if (slot_temp > slot_end - 3) {
                 return 1;
               }
               else {
                 if ((blocks[index].block.tags[slot_temp + 1 - QUQU_PRESLOT] == 0)
                   && (blocks[index].block.tags[slot_temp + 2 - QUQU_PRESLOT] == 0)) {
                   int sum = 0;
                   for (uint64_t i = slot_check + 1; i <= slot_temp; i++) {
                     sum += blocks[index].block.tags[i - QUQU_PRESLOT];
                   }
                   return 3 + sum;
                 }
                 else {
                   return 1;
                 }
               }
             }
           }
           // could be short counter
           else {
             // counter
             if (  (blocks[index].block.tags[slot_temp + 1 - QUQU_PRESLOT] == 0)
               && (blocks[index].block.tags[slot_temp + 2 - QUQU_PRESLOT] == 0)) {
               return 3 + temp_tag;
             }
             // not counter
             else {
               return 1;
             }
           }
         }
       }
     }
     else {
      return false;
     }
   }
      // other tags than 0, 255
      else {
        while (check_indexes != 0) {
          slot_check = _tzcnt_u64(check_indexes);
          if (slot_check == slot_start) {
            //ultimatum >>>>>>>>>>
            if (slot_check == slot_end - 1) {
              return 1;
            }
            else {
              uint64_t slot_temp = slot_check + 1;
              uint8_t temp_tag = blocks[index].block.tags[slot_temp - QUQU_PRESLOT];
              if (temp_tag == 0) {
                check_indexes &= ~(one[0] << slot_check);
                slot_temp = _tzcnt_u64(check_indexes);
                if (slot_temp >= slot_end) {
                  printf("ERROR6\n");
                  return false; // This will not happen
                }
                else {
                  uint64_t slot_diff = slot_temp - slot_check - 3;
                  uint8_t ret_tag = blocks[index].block.tags[slot_temp - 1 - QUQU_PRESLOT];
                  if (slot_diff < 0) {
                    printf("ERROR7\n");
                    return false;
                  }
                  else {
                    if (ret_tag > tag) ret_tag--;
                      return 2 + (QUQU_MAX - 1)*slot_diff + ret_tag;
                  }
                }
              }
              else if (temp_tag < tag) {
                return 2 + temp_tag;
              }
              // equal to
              else if (temp_tag == tag) {
                return 2;
              }
              // greater than
              else {
                return 1;
              }
            }
            // ultimatum <<<<<<<<<<
          }
          else if (slot_check == slot_end - 1) {
            return 1;
          }
          else {
            if (blocks[index].block.tags[slot_check - 1 - QUQU_PRESLOT] > tag) {
            }
            else if (blocks[index].block.tags[slot_check - 1 - QUQU_PRESLOT] == 0) {
              if (slot_check == slot_start + 1) {
                if (slot_check < slot_end - 2) {
                  if (blocks[index].block.tags[slot_check + 1 - QUQU_PRESLOT] == 0
                    && blocks[index].block.tags[slot_check + 2 - QUQU_PRESLOT] == 0) {
                  }
                  else {
                    //ultimatum >>>>>>>>>>
                    if (slot_check == slot_end - 1) {
                      return 1;
                    }
                    else {
                      uint64_t slot_temp = slot_check + 1;
                      uint8_t temp_tag = blocks[index].block.tags[slot_temp - QUQU_PRESLOT];
                      if (temp_tag == 0) {
                        check_indexes &= ~(one[0] << slot_check);
                        slot_temp = _tzcnt_u64(check_indexes);
                        if (slot_temp >= slot_end) {
                          printf("ERROR\n");
                          return false; // This will not happen
                        }
                        else {
                          uint64_t slot_diff = slot_temp - slot_check - 3;
                          uint8_t ret_tag = blocks[index].block.tags[slot_temp - 1 - QUQU_PRESLOT];
                          if (slot_diff < 0) {
                            printf("ERROR\n");
                            return false;
                          }
                          else {
                            if (ret_tag > tag) ret_tag--;
                              return 2 + (QUQU_MAX - 1)*slot_diff + ret_tag;
                          }
                        }
                      }
                      else if (temp_tag < tag) {
                        return 2 + temp_tag;
                      }
                      // equal to
                      else if (temp_tag == tag) {
                        return 2;
                      }
                      // greater than
                      else {
                        return 1;
                      }
                    }
                    // ultimatum <<<<<<<<<<
                  }
                }
                else {
                  //ultimatum >>>>>>>>>>
                  if (slot_check == slot_end - 1) {
                    return 1;
                  }
                  else {
                    uint64_t slot_temp = slot_check + 1;
                    uint8_t temp_tag = blocks[index].block.tags[slot_temp - QUQU_PRESLOT];
                    if (temp_tag == 0) {
                      check_indexes &= ~(one[0] << slot_check);
                      slot_temp = _tzcnt_u64(check_indexes);
                      if (slot_temp >= slot_end) {
                        printf("ERROR\n");
                        return false; // This will not happen
                      }
                      else {
                        uint64_t slot_diff = slot_temp - slot_check - 3;
                        uint8_t ret_tag = blocks[index].block.tags[slot_temp - 1 - QUQU_PRESLOT];
                        if (slot_diff < 0) {
                          printf("ERROR\n");
                          return false;
                        }
                        else {
                          if (ret_tag > tag) ret_tag--;
                            return 2 + (QUQU_MAX - 1)*slot_diff + ret_tag;
                        }
                      }
                    }
                    else if (temp_tag < tag) {
                      return 2 + temp_tag;
                    }
                    // equal to
                    else if (temp_tag == tag) {
                      return 2;
                    }
                    // greater than
                    else {
                      return 1;
                    }
                  }
                  // ultimatum <<<<<<<<<<
                }
              }
              else {
                if (blocks[index].block.tags[slot_check - 2 - QUQU_PRESLOT] == 0) {
                //ultimatum >>>>>>>>>>
                  if (slot_check == slot_end - 1) {
                    return 1;
                  }
                  else {
                    uint64_t slot_temp = slot_check + 1;
                    uint8_t temp_tag = blocks[index].block.tags[slot_temp - QUQU_PRESLOT];
                    if (temp_tag == 0) {
                      check_indexes &= ~(one[0] << slot_check);
                      slot_temp = _tzcnt_u64(check_indexes);
                      if (slot_temp >= slot_end) {
                        printf("ERROR\n");
                        return false; // This will not happen
                      }
                      else {
                        uint64_t slot_diff = slot_temp - slot_check - 3;
                        uint8_t ret_tag = blocks[index].block.tags[slot_temp - 1 - QUQU_PRESLOT];
                        if (slot_diff < 0) {
                          printf("ERROR\n");
                          return false;
                        }
                        else {
                          if (ret_tag > tag) ret_tag--;
                          return 2 + (QUQU_MAX - 1)*slot_diff + ret_tag;
                        }
                      }
                    }
                    else if (temp_tag < tag) {
                      return 2 + temp_tag;
                    }
                    // equal to
                    else if (temp_tag == tag) {
                      return 2;
                    }
                    // greater than
                    else {
                      return 1;
                    }
                  }
                  // ultimatum <<<<<<<<<<
                }
                else {
                }
              }
            }
            else if (blocks[index].block.tags[slot_check - 1 - QUQU_PRESLOT] < tag) {
              //ultimatum >>>>>>>>>>
              if (slot_check == slot_end - 1) {
                return 1;
              }
              else {
                uint64_t slot_temp = slot_check + 1;
                uint8_t temp_tag = blocks[index].block.tags[slot_temp - QUQU_PRESLOT];
                if (temp_tag == 0) {
                  check_indexes &= ~(one[0] << slot_check);
                  slot_temp = _tzcnt_u64(check_indexes);
                  if (slot_temp >= slot_end) {
                    printf("ERROR\n");
                    return false; // This will not happen
                  }
                  else {
                    uint64_t slot_diff = slot_temp - slot_check - 3;
                    uint8_t ret_tag = blocks[index].block.tags[slot_temp - 1 - QUQU_PRESLOT];
                    if (slot_diff < 0) {
                      printf("ERROR\n");
                      return false;
                    }
                    else {
                      if (ret_tag > tag) ret_tag--;
                      return 2 + (QUQU_MAX - 1)*slot_diff + ret_tag;
                    }
                  }
                }
                else if (temp_tag < tag) {
                  return 2 + temp_tag;
                }
                // equal to
                else if (temp_tag == tag) {
                  return 2;
                }
                // greater than
                else {
                  return 1;
                }
              }
              // ultimatum <<<<<<<<<<
            }
            else {
            }
          }
          check_indexes &= ~(one[0] << slot_check);
        }
      }
      //return true;

      // if every matching tags are counters
      return 0;
   } else {
      return 0;
   }
}

// If the item goes in the i'th slot (starting from 0) in the block then
// select(i) - i is the slot index for the end of the run.
bool vqf_is_present(vqf_filter * restrict filter, uint64_t hash) { /*CYDBG return value bool ->int*/
   vqf_metadata * restrict metadata           = &filter->metadata;
   //vqf_block    * restrict blocks             = filter->blocks;
   uint64_t                 key_remainder_bits = metadata->key_remainder_bits;
   uint64_t                 range              = metadata->range;

   uint64_t block_index = hash >> key_remainder_bits;
   uint64_t tag = hash & TAG_MASK;
   uint64_t alt_block_index = ((hash ^ (tag * 0x5bd1e995)) % range) >> key_remainder_bits;

   __builtin_prefetch(&filter->blocks[alt_block_index / QUQU_BUCKETS_PER_BLOCK].block);

//   printf("[CYDBG] hash: %lx\n", hash);
   return check_tags(filter, tag, block_index) || check_tags(filter, tag, alt_block_index);
/*   if (!re) {
     printf("\nLOOKUP FAIL hash: %lx, tag: %ld,\n", hash, tag);
     print_block(filter, block_index / QUQU_BUCKETS_PER_BLOCK);
     print_block(filter, alt_block_index / QUQU_BUCKETS_PER_BLOCK);
   }
   return re;*/
}


int get_count(vqf_filter * restrict filter, uint64_t hash) {
   vqf_metadata * restrict metadata           = &filter->metadata;
   //vqf_block    * restrict blocks             = filter->blocks;
   uint64_t                 key_remainder_bits = metadata->key_remainder_bits;
   uint64_t                 range              = metadata->range;

   uint64_t block_index = hash >> key_remainder_bits;
   uint64_t tag = hash & TAG_MASK;
   uint64_t alt_block_index = ((hash ^ (tag * 0x5bd1e995)) % range) >> key_remainder_bits;

   __builtin_prefetch(&filter->blocks[alt_block_index / QUQU_BUCKETS_PER_BLOCK].block);

   if (block_index == alt_block_index)
      return count_tags(filter, tag, block_index);
   else
      return count_tags(filter, tag, block_index) + count_tags(filter, tag, alt_block_index);

   /*if (!ret) {*/
   /*printf("tag: %ld offset: %ld\n", tag, block_index % QUQU_SLOTS_PER_BLOCK);*/
   /*print_block(filter, block_index / QUQU_SLOTS_PER_BLOCK);*/
   /*print_block(filter, alt_block_index / QUQU_SLOTS_PER_BLOCK);*/
   /*}*/
}

bool check_space(vqf_filter* restrict filter, uint64_t tag, uint64_t block_index) {
  if (tag == 0) {
    uint64_t count0 = count_tags(filter, tag, block_index);
    if ((count0 >= QUQU_MAX + 3 && (count0 - (QUQU_MAX + 3)) % QUQU_MAX == 0) || count0 <= 3) {
      return false;
    } else {
      return true;
    }
  } else if (tag == QUQU_MAX) {
    uint64_t countmax = count_tags(filter, tag, block_index);
    if ((countmax >= QUQU_MAX + 1 && (countmax - 2) % (QUQU_MAX - 1) == 0) || countmax <= 2) {
      return false;
    } else {
      return true;
    }
  } else {
    uint64_t countn = count_tags(filter, tag, block_index);
    if (block_index / QUQU_BUCKETS_PER_BLOCK == 987) {
      printf("[CYDBG] tag: %ld, countn: %ld\n", tag, countn);
    }
    if ((countn >= QUQU_MAX + 1 && (countn - 2) % (QUQU_MAX - 1) == 0) || countn <= 2) {
      return false;
    } else {
      return true;
    }
  }
}

vqf_block* add_block(vqf_filter * restrict filter, uint64_t block_index) {
  linked_blocks *new_block;
  new_block = (linked_blocks *)malloc(sizeof(linked_blocks));
  linked_blocks * restrict blocks = filter->blocks;
#if TAG_BITS == 8
   new_block->block.md[0] = UINT64_MAX;
   new_block->block.md[1] = UINT64_MAX;
   new_block->next = NULL;
#elif TAG_BITS == 16
   new_block->block.md = UINT64_MAX;
   new_block->next = NULL;
#endif
  printf("--------add_block--------\n");
  printf("block index: %ld\n", block_index);
  printf("metadata: ");
  uint64_t *md = new_block->block.md;
  print_bits(*(__uint128_t *)md, QUQU_BUCKETS_PER_BLOCK + QUQU_SLOTS_PER_BLOCK);
  printf("tags: ");
  print_tags(new_block->block.tags, QUQU_SLOTS_PER_BLOCK);

  blocks[block_index].next = new_block;
/*  lock(blocks[block_index/QUQU_BUCKETS_PER_BLOCK].block);
#if TAG_BITS == 8
  uint64_t *block_md = blocks[block_index/QUQU_BUCKETS_PER_BLOCK].block.md;
  uint64_t block_free = get_block_free_space(block_md);
#elif TAG_BITS == 16
  uint64_t *block_md = &blocks[block_index/QUQU_BUCKETS_PER_BLOCK].block.md;
  uint64_t block_free = get_block_free_space(*block_md);
#endif
  uint64_t tag = hash & TAG_MASK;

  blocks[block_index/QUQU_BUCKETS_PER_BLOCK].next = new_block;
  
  printf("[CYDBG] new_block size: %ld\n", sizeof(new_block));
  printf("[CYDBG] linked_blocks size: %ld\n", sizeof(linked_blocks));
  printf("[CYDBG] vqf_block size: %ld\n", sizeof(vqf_block));*/

  return &new_block->block;
}
