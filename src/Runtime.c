// HVM2 Core: single-thread, polarized, LAM/APP & DUP/SUP only

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__SSE__) || defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
#include <emmintrin.h> // SSE2
#include <immintrin.h> // AVX and newer
#define HVM_HAS_SIMD 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HVM_HAS_SIMD 1
#else
#define HVM_HAS_SIMD 0
#endif

#define DEBUG_LOG(fmt, ...) printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)

// WINDOWS WIP
#if defined(WIN16) || defined(WIN32) || defined(WIN64) || defined(_WIN16) ||   \
    defined(_WIN32) || defined(_WIN64) || defined(__TOS_WIN__) ||              \
    defined(__WIN16) || defined(__WIN16__) || defined(__WIN32) ||              \
    defined(__WIN32__) || defined(__WIN64) || defined(__WIN64__) ||            \
    defined(__WINDOWS__)

#include <windows.h>

typedef HANDLE thread_t;

int get_num_threads() {
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  return sysinfo.dwNumberOfProcessors;
}

typedef void *(*thread_func)(void *);

int thread_create(thread_t *thread, void *attr, thread_func start_routine,
                  void *arg) {
  if (attr != NULL) {
    printf("Ignoring thread attributes in Windows version\n");
  }

  *thread =
      CreateThread(NULL, // Default security attributes
                   0,    // Default stack size
                   (LPTHREAD_START_ROUTINE)start_routine, // Thread function
                   arg, // Thread function argument
                   0,   // Creation flags (0 = run immediately)
                   NULL // Thread ID (optional, can be NULL)
      );

  if (*thread == NULL) {
    return GetLastError();
  }

  return 0;
}

int thread_join(HANDLE thread, void **retval) {
  DWORD wait_result = WaitForSingleObject(thread, INFINITE);

  if (wait_result == WAIT_FAILED) {
    return GetLastError();
  }

  if (retval != NULL) {
    DWORD exit_code;
    if (GetExitCodeThread(thread, &exit_code)) {
      *retval = (void *)(intptr_t)exit_code;
    }
  }

  CloseHandle(thread);

  return 0;
}

#else

#include <pthread.h>
#include <unistd.h>

typedef pthread_t thread_t;

int get_num_threads() {
  long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
  return (nprocs < 1) ? 1 : nprocs;
}

int thread_create(pthread_t *thread, const pthread_attr_t *attr,
                  void *(*start_routine)(void *), void *arg) {
  return pthread_create(thread, attr, start_routine, arg);
}

int thread_join(pthread_t thread, void **retval) {
  return pthread_join(thread, retval);
}

#endif

#define MAX_THREADS get_num_threads() // uncomment for multi-threading

/*int MAX_THREADS = 1; // uncomment for testing*/

typedef uint8_t Tag;   //  8 bits
typedef uint32_t Lab;  // 24 bits
typedef uint32_t Loc;  // 32 bits
typedef uint64_t Term; // Loc | Lab | Tag
typedef uint32_t u32;
typedef uint64_t u64;
typedef int32_t i32;
typedef float f32;

// Constants
#define VAR 0x01
#define SUB 0x02
#define NUL 0x03
#define ERA 0x04
#define LAM 0x05
#define APP 0x06
#define SUP 0x07
#define DUP 0x08
#define REF 0x09
#define OPX 0x0A
#define OPY 0x0B
#define U32 0x0C
#define I32 0x0D
#define F32 0x0E
#define MAT 0x0F

const Term VOID = 0;

// Operators
#define OP_ADD 0x00
#define OP_SUB 0x01
#define OP_MUL 0x02
#define OP_DIV 0x03
#define OP_MOD 0x04
#define OP_EQ 0x05
#define OP_NE 0x06
#define OP_LT 0x07
#define OP_GT 0x08
#define OP_LTE 0x09
#define OP_GTE 0x0A
#define OP_AND 0x0B
#define OP_OR 0x0C
#define OP_XOR 0x0D
#define OP_LSH 0x0E
#define OP_RSH 0x0F

// Types
typedef uint64_t u64;
typedef _Atomic(u64) a64;

// Using union for type punning (safer in C)
typedef union {
    u32 u;
    i32 i;
    f32 f;
} TypeConverter;

// Global heap
static a64 *BUFF = NULL;
static a64 RNOD_INI = 0;
static a64 RNOD_END = 0;
static a64 RBAG = 0x1000000;
static a64 RBAG_INI = 0;
static a64 RBAG_END = 0;

// Global book
typedef struct Def {
  char *name;
  Term *nodes;
  u64 nodes_len;
  Term *rbag;
  u64 rbag_len;
} Def;

typedef struct Book {
  Def *defs;
  u32 len;
  u32 cap;
} Book;

static Book BOOK = {
    .defs = NULL,
    .len = 0,
    .cap = 0,
};

// Debugging
static char *tag_to_str(Tag tag);
void dump_buff();

// Term operations
Term term_new(Tag tag, Lab lab, Loc loc) {
  return ((Term)loc << 32) | ((Term)lab << 8) | tag;
}

Tag term_tag(Term term) { return term & 0xFF; }

Lab term_lab(Term term) { return (term >> 8) & 0xFFFFFF; }

Loc term_loc(Term term) { return (term >> 32) & 0xFFFFFFFF; }

Term term_offset_loc(Term term, Loc offset) {
  Tag tag = term_tag(term);
  if (tag == SUB || tag == NUL || tag == ERA || tag == REF || tag == U32) {
    return term;
  }

  Loc loc = term_loc(term) + offset;

  return (term & 0xFFFFFFFF) | (((Term)loc) << 32);
}

// Memory operations
Term swap(Loc loc, Term term) {
  return atomic_exchange_explicit(&BUFF[loc], term, memory_order_relaxed);
}

Term get(Loc loc) {
  return atomic_load_explicit(&BUFF[loc], memory_order_relaxed);
}

Term take(Loc loc) {
  return atomic_exchange_explicit(&BUFF[loc], VOID, memory_order_relaxed);
}

void set(Loc loc, Term term) {
  atomic_store_explicit(&BUFF[loc], term, memory_order_relaxed);
}

Loc port(u64 n, Loc x) { return n + x - 1; }

// Allocation
Loc alloc_node(u64 arity) {
  Loc loc = atomic_fetch_add(&RNOD_END, arity);
  return loc;
}

u64 inc_itr() {
  return atomic_load(&RNOD_END) / 2;
} // if (atomic_load(&RNOD_END) % 2 == 0) { return atomic_load(&RNOD_END) / 2; }
  // }

#if HVM_HAS_SIMD
// Copy a block of Terms using SIMD
void simd_copy_terms(Term *dst, const Term *src, size_t count) {
#if defined(__AVX512F__)
    // AVX-512 version - 64 bytes at a time (8 Terms)
    for (size_t i = 0; i < count / 8; i++) {
        __m512i data = _mm512_loadu_si512((__m512i*)&src[i*8]);
        _mm512_storeu_si512((__m512i*)&dst[i*8], data);
    }
    
    // Handle remainder
    for (size_t i = (count / 8) * 8; i < count; i++) {
        dst[i] = src[i];
    }
#elif defined(__AVX2__)
    // AVX2 version - 32 bytes at a time (4 Terms)
    for (size_t i = 0; i < count / 4; i++) {
        __m256i data = _mm256_loadu_si256((__m256i*)&src[i*4]);
        _mm256_storeu_si256((__m256i*)&dst[i*4], data);
    }
    
    // Handle remainder
    for (size_t i = (count / 4) * 4; i < count; i++) {
        dst[i] = src[i];
    }
#elif defined(__SSE2__)
    // SSE2 version - 16 bytes at a time (2 Terms)
    for (size_t i = 0; i < count / 2; i++) {
        __m128i data = _mm_loadu_si128((__m128i*)&src[i*2]);
        _mm_storeu_si128((__m128i*)&dst[i*2], data);
    }
    
    // Handle remainder
    if (count % 2) {
        dst[count-1] = src[count-1];
    }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    // NEON version - 16 bytes at a time (2 Terms)
    for (size_t i = 0; i < count / 2; i++) {
        uint64x2_t data = vld1q_u64((const uint64_t*)&src[i*2]);
        vst1q_u64((uint64_t*)&dst[i*2], data);
    }
    
    // Handle remainder
    if (count % 2) {
        dst[count-1] = src[count-1];
    }
#else
    // Fallback for non-SIMD architectures
    memcpy(dst, src, count * sizeof(Term));
#endif
}
#endif


Loc rbag_push(Term neg, Term pos) {
  // Atomically fetch and add to increment RBAG_END
  // This ensures thread-safe allocation of a new location pair
  u64 current_end = atomic_fetch_add(&RBAG_END, 2);

  // Calculate the absolute location for this pair
  Loc loc = RBAG + current_end;

  // Set the negative and positive terms
  atomic_store_explicit(&BUFF[loc + 0], neg, memory_order_release);
  atomic_store_explicit(&BUFF[loc + 1], pos, memory_order_release);

  // Return the location of the newly pushed pair
  return loc;
}

Loc rbag_pop() {
  // Use atomic fetch_add to increment RBAG_INI atomically
  // This ensures thread-safe access to the reduction bag
  u64 current_ini = atomic_fetch_add(&RBAG_INI, 2);
  u64 current_end = atomic_load_explicit(&RBAG_END, memory_order_acquire);

  // Check if we've reached or exceeded the end of the reduction bag
  if (current_ini >= current_end) {
    // If we've exhausted the reduction bag, return 0
    return 0;
  }

  // Calculate and return the location
  // current_ini represents the starting index of the location pair
  return RBAG + current_ini;
}

Loc rbag_ini() {
  // Atomically load the current initial index of the reduction bag
  return RBAG + atomic_load(&RBAG_INI);
}

Loc rbag_end() {
  // Atomically load the current end index of the reduction bag
  return RBAG + atomic_load(&RBAG_END);
}

Loc rnod_end() {
  // Atomically load the current end of the reduction nodes
  return atomic_load(&RNOD_END);
}

// Book operations

// Moves the global buffer and redex bag into a new def and resets
// the global buffer and redex bag.
void def_new(char *name) {
  if (BOOK.len == BOOK.cap) {
    if (BOOK.cap == 0) {
      BOOK.cap = 32;
    } else {
      BOOK.cap *= 2;
    }

    BOOK.defs = realloc(BOOK.defs, sizeof(Def) * BOOK.cap);
  }

  u64 rbag_end = atomic_load(&RBAG_END);
  u64 rnod_end = atomic_load(&RNOD_END);

  Def def = {
      .name = name,
      .nodes = calloc(rnod_end, sizeof(Term)),
      .nodes_len = rnod_end,
      .rbag = calloc(rbag_end, sizeof(Term)),
      .rbag_len = rbag_end,
  };

#if HVM_HAS_SIMD
  // Use SIMD operations for copying
  simd_copy_terms(def.nodes, (Term*)BUFF, def.nodes_len);
  simd_copy_terms(def.rbag, (Term*)(BUFF + RBAG), def.rbag_len);
#else
  memcpy(def.nodes, BUFF, sizeof(Term) * def.nodes_len);
  memcpy(def.rbag, BUFF + RBAG, sizeof(Term) * def.rbag_len);
#endif

  // Clear the buffers
  memset(BUFF, 0, sizeof(Term) * def.nodes_len);
  memset(BUFF + RBAG, 0, sizeof(Term) * def.rbag_len);

  atomic_store(&RNOD_END, 0);
  atomic_store(&RBAG_END, 0);

  BOOK.defs[BOOK.len] = def;
  BOOK.len++;
}

char *def_name(Loc def_idx) { return BOOK.defs[def_idx].name; }

// Expands a ref's data into a linear block of nodes with its nodes' locs
// offset by the index where in the BUFF it was expanded.
//
// Returns the ref's root, the first node in its data.
Term expand_ref(Loc def_idx) {
  if (RNOD_END == 0) {
    printf("expand_ref: empty BUFF\n");
    exit(1);
  }

  Def def = BOOK.defs[def_idx];
  const u32 nodes_len = def.nodes_len;
  const Term *nodes = def.nodes;
  const Term *rbag = def.rbag;
  const u32 rbag_len = def.rbag_len;

  a64 offset = atomic_fetch_add(&RNOD_END, nodes_len - 1) - 1;

  Term root = term_offset_loc(nodes[0], offset);

#if HVM_HAS_SIMD && defined(__SSE2__)
  // Optimize for SIMD if available and nodes have significant size
  if (nodes_len > 16) {
    u32 i = 1;
    
    // Process 8 terms at a time using SIMD
    for (; i + 7 < nodes_len; i += 8) {
      for (int j = 0; j < 8; j++) {
        set(i + offset + j, term_offset_loc(nodes[i + j], offset));
      }
    }
    
    // Process remaining terms
    for (; i < nodes_len; i++) {
      set(i + offset, term_offset_loc(nodes[i], offset));
    }
  } else {
    // For small node lists, use the original approach
    u32 i = 1;
    for (; i < nodes_len; i++) {
      set(i + offset, term_offset_loc(nodes[i], offset));
    }
  }
#else
  // Unroll loop, better branch prediction
  u32 i = 1;
  for (; i + 7 < nodes_len; i += 8) {
    set(i + offset, term_offset_loc(nodes[i], offset));
    set(i + offset + 1, term_offset_loc(nodes[i + 1], offset));
    set(i + offset + 2, term_offset_loc(nodes[i + 2], offset));
    set(i + offset + 3, term_offset_loc(nodes[i + 3], offset));
    set(i + offset + 4, term_offset_loc(nodes[i + 4], offset));
    set(i + offset + 5, term_offset_loc(nodes[i + 5], offset));
    set(i + offset + 6, term_offset_loc(nodes[i + 6], offset));
    set(i + offset + 7, term_offset_loc(nodes[i + 7], offset));
  }

  // Remaining nodes
  for (; i < nodes_len; i++) {
    set(i + offset, term_offset_loc(nodes[i], offset));
  }
#endif

  // Redexes in batches of 2 (already aligned)
  for (u32 i = 0; i < rbag_len; i += 2) {
    rbag_push(term_offset_loc(rbag[i], offset),
              term_offset_loc(rbag[i + 1], offset));
  }

  return root;
}

// Atomic Linker
static inline void move(Loc neg_loc, u64 pos);

static inline void link_terms(Term neg, Term pos) {
  if (term_tag(pos) == VAR) {
    Term far = swap(term_loc(pos), neg);
    if (term_tag(far) != SUB) {
      move(term_loc(pos), far);
    }
  } else {
    rbag_push(neg, pos);
  }
}

static inline void move(Loc neg_loc, Term pos) {
  Term neg = swap(neg_loc, pos);
  if (term_tag(neg) != SUB) {
    // No need to take() since we already swapped
    link_terms(neg, pos);
  }
}

// Interactions
static void interact_applam(Loc a_loc, Loc b_loc) {
  Term arg = take(port(1, a_loc));
  Loc ret = port(2, a_loc);
  Loc var = port(1, b_loc);
  Term bod = take(port(2, b_loc));
  move(var, arg);
  move(ret, bod);
}

static void interact_appsup(Loc a_loc, Loc b_loc) {
  Term arg = take(port(1, a_loc));
  Loc ret = port(2, a_loc);
  Term tm1 = take(port(1, b_loc));
  Term tm2 = take(port(2, b_loc));
  Loc dp1 = alloc_node(2);
  Loc dp2 = alloc_node(2);
  Loc cn1 = alloc_node(2);
  Loc cn2 = alloc_node(2);
  set(port(1, dp1), term_new(SUB, 0, 0));
  set(port(2, dp1), term_new(SUB, 0, 0));
  set(port(1, dp2), term_new(VAR, 0, port(2, cn1)));
  set(port(2, dp2), term_new(VAR, 0, port(2, cn2)));
  set(port(1, cn1), term_new(VAR, 0, port(1, dp1)));
  set(port(2, cn1), term_new(SUB, 0, 0));
  set(port(1, cn2), term_new(VAR, 0, port(2, dp1)));
  set(port(2, cn2), term_new(SUB, 0, 0));
  link_terms(term_new(DUP, 0, dp1), arg);
  move(ret, term_new(SUP, 0, dp2));
  link_terms(term_new(APP, 0, cn1), tm1);
  link_terms(term_new(APP, 0, cn2), tm2);
}

static void interact_appnul(Loc a_loc) {
  Term arg = take(port(1, a_loc));
  Loc ret = port(2, a_loc);
  link_terms(term_new(ERA, 0, 0), arg);
  move(ret, term_new(NUL, 0, 0));
}

static void interact_appu32(Loc a_loc, u32 num) {
  Term arg = take(port(1, a_loc));
  Loc ret = port(2, a_loc);
  link_terms(term_new(U32, 0, num), arg);
  move(ret, term_new(U32, 0, num));
}

static void interact_opxnul(Loc a_loc) {
  Term arg = take(port(1, a_loc));
  Loc ret = port(2, a_loc);
  link_terms(term_new(ERA, 0, 0), arg);
  move(ret, term_new(NUL, 0, 0));
}

static void interact_opxnum(Loc a_loc, Lab op, u32 num, Tag num_type) {
  Term arg = swap(port(1, a_loc), term_new(num_type, 0, num));
  link_terms(term_new(OPY, op, a_loc), arg);
}

static void interact_opxsup(Loc a_loc, Lab op, Loc b_loc) {
  Term arg = take(port(1, a_loc));
  Loc ret = port(2, a_loc);
  Term tm1 = take(port(1, b_loc));
  Term tm2 = take(port(2, b_loc));
  Loc dp1 = alloc_node(2);
  Loc dp2 = alloc_node(2);
  Loc cn1 = alloc_node(2);
  Loc cn2 = alloc_node(2);
  set(port(1, dp1), term_new(SUB, 0, 0));
  set(port(2, dp1), term_new(SUB, 0, 0));
  set(port(1, dp2), term_new(VAR, 0, port(2, cn1)));
  set(port(2, dp2), term_new(VAR, 0, port(2, cn2)));
  set(port(1, cn1), term_new(VAR, 0, port(1, dp1)));
  set(port(2, cn1), term_new(SUB, 0, 0));
  set(port(1, cn2), term_new(VAR, 0, port(2, dp1)));
  set(port(2, cn2), term_new(SUB, 0, 0));
  link_terms(term_new(DUP, 0, dp1), arg);
  move(ret, term_new(SUP, 0, dp2));
  link_terms(term_new(OPX, op, cn1), tm1);
  link_terms(term_new(OPX, op, cn2), tm2);
}

static void interact_opynul(Loc a_loc) {
  Term arg = take(port(1, a_loc));
  Loc ret = port(2, a_loc);
  link_terms(term_new(ERA, 0, 0), arg);
  move(ret, term_new(NUL, 0, 0));
}

// Safer Utilities
u32 u32_to_u32(u32 u) { return u; }  

i32 u32_to_i32(u32 u) { 
    TypeConverter converter;
    converter.u = u;
    return converter.i;
}

f32 u32_to_f32(u32 u) {
    TypeConverter converter;
    converter.u = u;
    return converter.f;
}

u32 i32_to_u32(i32 i) {
    TypeConverter converter;
    converter.i = i;
    return converter.u;
}

u32 f32_to_u32(f32 f) {
    TypeConverter converter;
    converter.f = f;
    return converter.u;
}

static void interact_opynum(Loc a_loc, Lab op, u32 y, Tag y_type) {
  u32 x = term_loc(take(port(1, a_loc)));
  Loc ret = port(2, a_loc);
  u32 res = 0;

  // Optimized path using jump table & direct compute
  if (y_type == U32) {
    // Faster operation dispatch
    static void *op_jumptable[] = {
        [OP_ADD] = &&do_add, [OP_SUB] = &&do_sub, [OP_MUL] = &&do_mul,
        [OP_DIV] = &&do_div, [OP_EQ] = &&do_eq,   [OP_NE] = &&do_ne,
        [OP_LT] = &&do_lt,   [OP_GT] = &&do_gt,   [OP_LTE] = &&do_lte,
        [OP_GTE] = &&do_gte, [OP_MOD] = &&do_mod, [OP_AND] = &&do_and,
        [OP_OR] = &&do_or,   [OP_XOR] = &&do_xor, [OP_LSH] = &&do_lsh,
        [OP_RSH] = &&do_rsh};

    // Faster branching
    goto *op_jumptable[op];

  do_add:
    res = x + y;
    goto done;
  do_sub:
    res = x - y;
    goto done;
  do_mul:
    res = x * y;
    goto done;
  do_div:
    res = x / y;
    goto done;
  do_eq:
    res = x == y;
    goto done;
  do_ne:
    res = x != y;
    goto done;
  do_lt:
    res = x < y;
    goto done;
  do_gt:
    res = x > y;
    goto done;
  do_lte:
    res = x <= y;
    goto done;
  do_gte:
    res = x >= y;
    goto done;
  do_mod:
    res = x % y;
    goto done;
  do_and:
    res = x & y;
    goto done;
  do_or:
    res = x | y;
    goto done;
  do_xor:
    res = x ^ y;
    goto done;
  do_lsh:
    res = x << y;
    goto done;
  do_rsh:
    res = x >> y;
    goto done;

  done:;
  } else {
    // Inlined type conversion and operation for i32 and f32
    switch (y_type) {
    case I32: {
      i32 a = u32_to_i32(x);
      i32 b = u32_to_i32(y);
      i32 val;
      switch (op) {
      case OP_ADD:
        val = a + b;
        break;
      case OP_SUB:
        val = a - b;
        break;
      case OP_MUL:
        val = a * b;
        break;
      case OP_DIV:
        val = a / b;
        break;
      case OP_EQ:
        val = a == b;
        break;
      case OP_NE:
        val = a != b;
        break;
      case OP_LT:
        val = a < b;
        break;
      case OP_GT:
        val = a > b;
        break;
      case OP_LTE:
        val = a <= b;
        break;
      case OP_GTE:
        val = a >= b;
        break;
      case OP_MOD:
        val = a % b;
        break;
      case OP_AND:
        val = a & b;
        break;
      case OP_OR:
        val = a | b;
        break;
      case OP_XOR:
        val = a ^ b;
        break;
      case OP_LSH:
        val = a << b;
        break;
      case OP_RSH:
        val = a >> b;
        break;
      default:
        val = 0;
      }
      res = i32_to_u32(val);
      break;
    }
    case F32: {
      f32 a = u32_to_f32(x);
      f32 b = u32_to_f32(y);
      f32 val;
      switch (op) {
      case OP_ADD:
        val = a + b;
        break;
      case OP_SUB:
        val = a - b;
        break;
      case OP_MUL:
        val = a * b;
        break;
      case OP_DIV:
        val = a / b;
        break;
      case OP_EQ:
        val = a == b;
        break;
      case OP_NE:
        val = a != b;
        break;
      case OP_LT:
        val = a < b;
        break;
      case OP_GT:
        val = a > b;
        break;
      case OP_LTE:
        val = a <= b;
        break;
      case OP_GTE:
        val = a >= b;
        break;
      default:
        val = 0;
      }
      res = f32_to_u32(val);
      break;
    }
    }
  }
  move(ret, term_new(y_type, 0, res));
}

static void interact_opysup(Loc a_loc, Loc b_loc) {
  Term arg = take(port(1, a_loc));
  Loc ret = port(2, a_loc);
  Term tm1 = take(port(1, b_loc));
  Term tm2 = take(port(2, b_loc));
  Loc dp1 = alloc_node(2);
  Loc dp2 = alloc_node(2);
  Loc cn1 = alloc_node(2);
  Loc cn2 = alloc_node(2);
  set(port(1, dp1), term_new(SUB, 0, 0));
  set(port(2, dp1), term_new(SUB, 0, 0));
  set(port(1, dp2), term_new(VAR, 0, port(2, cn1)));
  set(port(2, dp2), term_new(VAR, 0, port(2, cn2)));
  set(port(1, cn1), term_new(VAR, 0, port(1, dp1)));
  set(port(2, cn1), term_new(SUB, 0, 0));
  set(port(1, cn2), term_new(VAR, 0, port(2, dp1)));
  set(port(2, cn2), term_new(SUB, 0, 0));
  link_terms(term_new(DUP, 0, dp1), arg);
  move(ret, term_new(SUP, 0, dp2));
  link_terms(term_new(OPY, 0, cn1), tm1);
  link_terms(term_new(OPY, 0, cn2), tm2);
}

static void interact_dupsup(Loc a_loc, Loc b_loc) {
  Loc dp1 = port(1, a_loc);
  Loc dp2 = port(2, a_loc);
  Term tm1 = take(port(1, b_loc));
  Term tm2 = take(port(2, b_loc));
  move(dp1, tm1);
  move(dp2, tm2);
}

static void interact_duplam(Loc a_loc, Loc b_loc) {
  Loc dp1 = port(1, a_loc);
  Loc dp2 = port(2, a_loc);
  Loc var = port(1, b_loc);
  // TODO(enricozb): why is this the only take?
  Term bod = take(port(2, b_loc));
  Loc co1 = alloc_node(2);
  Loc co2 = alloc_node(2);
  Loc du1 = alloc_node(2);
  Loc du2 = alloc_node(2);
  set(port(1, co1), term_new(SUB, 0, 0));
  set(port(2, co1), term_new(VAR, 0, port(1, du2)));
  set(port(1, co2), term_new(SUB, 0, 0));
  set(port(2, co2), term_new(VAR, 0, port(2, du2)));
  set(port(1, du1), term_new(VAR, 0, port(1, co1)));
  set(port(2, du1), term_new(VAR, 0, port(1, co2)));
  set(port(1, du2), term_new(SUB, 0, 0));
  set(port(2, du2), term_new(SUB, 0, 0));
  move(dp1, term_new(LAM, 0, co1));
  move(dp2, term_new(LAM, 0, co2));
  move(var, term_new(SUP, 0, du1));
  link_terms(term_new(DUP, 0, du2), bod);
}

static void interact_dupnul(Loc a_loc) {
  Loc dp1 = port(1, a_loc);
  Loc dp2 = port(2, a_loc);
  move(dp1, term_new(NUL, 0, a_loc));
  move(dp2, term_new(NUL, 0, a_loc));
}

static void interact_dupnum(Loc a_loc, u32 n, Tag n_type) {
  Loc dp1 = port(1, a_loc);
  Loc dp2 = port(2, a_loc);
  move(dp1, term_new(n_type, 0, n));
  move(dp2, term_new(n_type, 0, n));
}

static void interact_dupref(Loc a_loc, Loc b_loc) {
  move(port(1, a_loc), term_new(REF, 0, b_loc));
  move(port(2, a_loc), term_new(REF, 0, b_loc));
}

static void interact_matnul(Loc a_loc, Lab mat_len) {
  move(port(1, a_loc), term_new(NUL, 0, 0));
  for (u32 i = 0; i < mat_len; i++) {
    link_terms(term_new(ERA, 0, 0), take(port(i + 2, a_loc)));
  }
}

static void interact_matnum(Loc mat_loc, Lab mat_len, u32 n, Tag n_type) {
  if (n_type != U32) {
    printf("match with non-U32\n");
    exit(1);
  }

  u32 i_arm = (n < mat_len - 1) ? n : (mat_len - 1);
  for (u32 i = 0; i < mat_len; i++) {
    if (i != i_arm) {
      link_terms(term_new(ERA, 0, 0), take(port(2 + i, mat_loc)));
    }
  }

  Loc ret = port(1, mat_loc);
  Term arm = take(port(2 + i_arm, mat_loc));

  if (i_arm < mat_len - 1) {
    move(ret, arm);
  } else {
    Loc app = alloc_node(2);
    set(app + 0, term_new(U32, 0, n - (mat_len - 1)));
    set(app + 1, term_new(SUB, 0, 0));
    move(ret, term_new(VAR, 0, port(2, app)));

    link_terms(term_new(APP, 0, app), arm);
  }
}

static void interact_matsup(Loc mat_loc, Lab mat_len, Loc sup_loc) {
  Loc ma0 = alloc_node(1 + mat_len);
  Loc ma1 = alloc_node(1 + mat_len);

  Loc sup = alloc_node(2);

  set(port(1, sup), term_new(VAR, 0, port(1, ma1)));
  set(port(2, sup), term_new(VAR, 0, port(1, ma0)));
  set(port(1, ma0), term_new(SUB, 0, 0));
  set(port(1, ma1), term_new(SUB, 0, 0));

  for (u64 i = 0; i < mat_len; i++) {
    Loc dui = alloc_node(2);
    set(port(1, dui), term_new(SUB, 0, 0));
    set(port(2, dui), term_new(SUB, 0, 0));
    set(port(2 + i, ma0), term_new(VAR, 0, port(2, dui)));
    set(port(2 + i, ma1), term_new(VAR, 0, port(1, dui)));

    link_terms(term_new(DUP, 0, dui), take(port(2 + i, mat_loc)));
  }

  move(port(1, mat_loc), term_new(SUP, 0, sup));
  link_terms(term_new(MAT, mat_len, ma0), take(port(2, sup_loc)));
  link_terms(term_new(MAT, mat_len, ma1), take(port(1, sup_loc)));
}

static void interact_eralam(Loc b_loc) {
  Loc var = port(1, b_loc);
  Term bod = take(port(2, b_loc));
  move(var, term_new(NUL, 0, 0));
  link_terms(term_new(ERA, 0, 0), bod);
}

static void interact_erasup(Loc b_loc) {
  Term tm1 = take(port(1, b_loc));
  Term tm2 = take(port(2, b_loc));
  link_terms(term_new(ERA, 0, 0), tm1);
  link_terms(term_new(ERA, 0, 0), tm2);
}

static char *tag_to_str(Tag tag);

static void interact(Term neg, Term pos) {
  Tag neg_tag = term_tag(neg);
  Tag pos_tag = term_tag(pos);
  Loc neg_loc = term_loc(neg);
  Loc pos_loc = term_loc(pos);

  switch (neg_tag) {
  case APP:
    switch (pos_tag) {
    case LAM:
      interact_applam(neg_loc, pos_loc);
      break;
    case NUL:
      interact_appnul(neg_loc);
      break;
    case U32:
      interact_appu32(neg_loc, pos_loc);
      break;
    case REF:
      link_terms(neg, expand_ref(pos_loc));
      break;
    case SUP:
      interact_appsup(neg_loc, pos_loc);
      break;
    }
    break;
  case OPX:
    switch (pos_tag) {
    case LAM:
      break;
    case NUL:
      interact_opxnul(neg_loc);
      break;
    case U32:
    case I32:
    case F32:
      interact_opxnum(neg_loc, term_lab(neg), pos_loc, pos_tag);
      break;
    case REF:
      link_terms(neg, expand_ref(pos_loc));
      break;
    case SUP:
      interact_opxsup(neg_loc, term_lab(neg), pos_loc);
      break;
    }
    break;
  case OPY:
    switch (pos_tag) {
    case LAM:
      break;
    case NUL:
      interact_opynul(neg_loc);
      break;
    case U32:
    case I32:
    case F32:
      interact_opynum(neg_loc, term_lab(neg), pos_loc, pos_tag);
      break;
    case REF:
      link_terms(neg, expand_ref(pos_loc));
      break;
    case SUP:
      interact_opysup(neg_loc, pos_loc);
      break;
    }
    break;
  case DUP:
    switch (pos_tag) {
    case LAM:
      interact_duplam(neg_loc, pos_loc);
      break;
    case NUL:
      interact_dupnul(neg_loc);
      break;
    case U32:
    case I32:
    case F32:
      interact_dupnum(neg_loc, pos_loc, pos_tag);
      break;
    // TODO(enricozb): dup-ref optimization
    case REF:
      interact_dupref(neg_loc, pos_loc);
      break;
    // case REF: link_terms(neg, expand_ref(pos_loc)); break;
    case SUP:
      interact_dupsup(neg_loc, pos_loc);
      break;
    }
    break;
  case MAT:
    switch (pos_tag) {
    case LAM:
      break;
    case NUL:
      interact_matnul(neg_loc, term_lab(neg));
      break;
    case U32:
    case I32:
    case F32:
      interact_matnum(neg_loc, term_lab(neg), pos_loc, pos_tag);
      break;
    case REF:
      link_terms(neg, expand_ref(pos_loc));
      break;
    case SUP:
      interact_matsup(neg_loc, term_lab(neg), pos_loc);
      break;
    }
    break;
  case ERA:
    switch (pos_tag) {
    case LAM:
      interact_eralam(pos_loc);
      break;
    case NUL:
      break;
    case U32:
      break;
    case REF:
      break;
    case SUP:
      interact_erasup(pos_loc);
      break;
    }
    break;
  }
}


static inline int thread_work() {
    Loc loc = rbag_pop();
   
    if(loc == 0) { return 0; }

    Term neg = take(loc);
    Term pos = take(loc + 1);
    interact(neg, pos);

    return 1;
}
    

void hvm_init() {
  if (BUFF == NULL) {
    BUFF = aligned_alloc(64, (1ULL << 26) * sizeof(a64));
    if (BUFF == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      exit(EXIT_FAILURE);
    }
  }
  memset(BUFF, 0, (1ULL << 26) * sizeof(a64)); // FIXED ALLOCATION

  atomic_store(&RNOD_INI, 0);
  atomic_store(&RNOD_END, 0);
  atomic_store(&RBAG_INI, 0);
  atomic_store(&RBAG_END, 0);
}

void hvm_free() {
  if (BUFF != NULL) {
    free(BUFF);
    BUFF = NULL;
  }
}

void boot(Loc def_idx) {
  if (RNOD_END || RBAG_END) {
    printf("booting on non-empty state\n");
    exit(1);
  }

  Loc root = alloc_node(1);
  set(root, expand_ref(def_idx));
}

Term normalize(Term term) {
  if (term_tag(term) != REF) {
    printf("normalizing non-ref\n");
    exit(1);
  }

  boot(term_loc(term));

  while (thread_work());  
  
  /*printf("MAX_THREADS: %u\n", MAX_THREADS);*/
  
  return get(0);
}

// Debugging
static char *tag_to_str(Tag tag) {
  switch (tag) {
  case VOID:
    return "___";
  case VAR:
    return "VAR";
  case SUB:
    return "SUB";
  case NUL:
    return "NUL";
  case ERA:
    return "ERA";
  case LAM:
    return "LAM";
  case APP:
    return "APP";
  case SUP:
    return "SUP";
  case DUP:
    return "DUP";
  case REF:
    return "REF";
  case OPX:
    return "OPX";
  case OPY:
    return "OPY";
  case U32:
    return "U32";
  case I32:
    return "I32";
  case F32:
    return "F32";
  case MAT:
    return "MAT";

  default:
    return "???";
  }
}

/*void dump_buff() {*/
/*  FILE *file = fopen("multi.txt", "w");*/
/*  if (file == NULL) {*/
/*    perror("Error opening file");*/
/*    return;*/
/*  }*/
/**/
/*  fprintf(file, "------------------\n");*/
/*  fprintf(file, "      NODES\n");*/
/*  fprintf(file, "ADDR   LOC LAB TAG\n");*/
/*  fprintf(file, "------------------\n");*/
/*  for (Loc loc = RNOD_INI; loc < RNOD_END; loc++) {*/
/*    Term term = get(loc);*/
/*    Loc t_loc = term_loc(term);*/
/*    Lab t_lab = term_lab(term);*/
/*    Tag t_tag = term_tag(term);*/
/*    fprintf(file, "%06X %03X %03X %s\n", loc, term_loc(term),
 * term_lab(term),*/
/*            tag_to_str(term_tag(term)));*/
/*  }*/
/**/
/*  fprintf(file, "------------------\n");*/
/*  fprintf(file, "    REDEX BAG\n");*/
/*  fprintf(file, "ADDR   LOC LAB TAG\n");*/
/*  fprintf(file, "------------------\n");*/
/*  for (Loc loc = RBAG + RBAG_INI; loc < RBAG + RBAG_END; loc++) {*/
/*    Term term = get(loc);*/
/*    Loc t_loc = term_loc(term);*/
/*    Lab t_lab = term_lab(term);*/
/*    Tag t_tag = term_tag(term);*/
/*    fprintf(file, "%06X %03X %03X %s\n", loc, term_loc(term),
 * term_lab(term),*/
/*            tag_to_str(term_tag(term)));*/
/*  }*/
/**/
/*  fprintf(file, "------------------\n");*/
/**/
/*  fclose(file);*/
/*}*/

void dump_buff() {
  printf("------------------\n");
  printf("      NODES\n");
  printf("ADDR   LOC LAB TAG\n");
  printf("------------------\n");
  for (Loc loc = RNOD_INI; loc < RNOD_END; loc++) {
    Term term = get(loc);
    Loc t_loc = term_loc(term);
    Lab t_lab = term_lab(term);
    Tag t_tag = term_tag(term);
    printf("%06X %03X %03X %s\n", loc, term_loc(term), term_lab(term),
           tag_to_str(term_tag(term)));
  }
  printf("------------------\n");
  printf("    REDEX BAG\n");
  printf("ADDR   LOC LAB TAG\n");
  printf("------------------\n");
  for (Loc loc = RBAG + RBAG_INI; loc < RBAG + RBAG_END; loc++) {
    Term term = get(loc);
    Loc t_loc = term_loc(term);
    Lab t_lab = term_lab(term);
    Tag t_tag = term_tag(term);
    printf("%06X %03X %03X %s\n", loc, term_loc(term), term_lab(term),
           tag_to_str(term_tag(term)));
  }
  printf("------------------\n");
}
