/*  Copyright 2022 StreamLogic, LLC.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include <map>
#include <list>
#include <vector>


static std::map<int,std::vector<int8_t> *> twiddles;

static const std::vector<int8_t>& get_twiddle(int N) {
  auto entry = twiddles.find(N);
  if (entry == twiddles.end()) {
    int halfPi = N/4;
    int count = N/2 + halfPi;
    double k = -6.283185307179586/N;
    auto tw = new std::vector<int8_t>();
    for (int i = 0; i < count; i++)
      tw->push_back((int)round(cos(i*k)*64));
    twiddles[N] = tw;
    return *tw;
  }
  return *(entry->second);
}

static int bitrev(int a, int bits) {
  uint16_t b = 0;
  for (int i = 0; i < bits; i++) {
    b = b << 1;
    if (a&1) b |= 1;
    a = a >> 1;
  }
  return b;
}

static uint32_t cpxrot(int8_t co, int16_t r, int8_t si, int16_t i) {
  int pr = co*r - si*i;
  int pi = si*r + co*i;
  int16_t fxr = pr>>6;
  int16_t fxi = pi>>6;
  uint32_t ret = (((uint32_t)fxr)<<16)|(uint16_t)fxi;
  return ret;
}

void rFFT16(const int N, const int M, const int16_t *win,
	    uint16_t *fft, int16_t *fr, int16_t *fi) {
  uint32_t count;
  uint32_t m;
  uint32_t k;
  uint32_t k0;
  uint32_t stp;
  uint32_t hi;
  uint32_t lo;
  uint32_t g;
  uint32_t j;
  uint32_t a1;
  uint32_t a2;
  int16_t samp;
  int16_t r1;
  int16_t i1;
  int16_t r2;
  int16_t i2;
  uint32_t prd;
  int16_t pr;
  int16_t pi;
  int8_t si;
  int8_t co;
  int16_t rout;
  int16_t iout;
  uint16_t mgr;
  uint16_t mgi;
  uint16_t mg;
  uint32_t wmsk = N-1;
  uint32_t hwmsk = (N/2)-1;

  const std::vector<int8_t>& twiddle=get_twiddle(N);

  for (count = 0; count < N; count++) {
    samp = win[count];
    //printf("[%d][%d] %d\n", bitrev(count, M), count, samp);
    fr[bitrev(count, M)] = samp;
    fi[count] = 0;
  }

  stp = N/2;
  lo = 0;
  hi = N-2;
  g = 1;
  m = 0;
  while (1) {
    k = 0;
    j = 0;
    while (1) {
      if (j == (N/2)) break;
      k0 = k;
      k = (k + stp) & hwmsk;
      a1 = ((j << 1) & hi) | (j & lo);
      a2 = a1 + g;
      j = j + 1;
      
      r2 = fr[a2];
      i2 = fi[a2];
      co = twiddle[k0];
      si = twiddle[(k0 + (N/4))];
      prd = cpxrot(co, r2, si, i2);
      pr = (int16_t)(prd >> 16) >> 1;
      pi = (int16_t)prd >> 1;
      
      r1 = fr[a1];
      i1 = fi[a1];
      fr[a1] = (r1 >> 1) + pr;
      fi[a1] = (i1 >> 1) + pi;
      fr[a2] = (r1 >> 1) - pr;
      fi[a2] = (i1 >> 1) - pi;
    }
    stp = stp >> 1;
    g = g << 1;
    lo = (lo << 1) | 1;
    hi = (hi << 1) & wmsk;
    m = m + 1;
    if (m == M) break;
  }

  for (count = 0; count < (N/2+1); count++) {
    rout = fr[count];
    iout = fi[count];
    //printf("%d + %di\n", rout, iout);
    if (rout >= 0)
      mgr = rout;
    else
      mgr = -rout;
    if (iout >= 0)
      mgi = iout;
    else
      mgi = -iout;
    mg = mgr + mgi;
    fft[count] = mg;
  }
}


static std::map<int,std::list<std::tuple<int,int,int,std::vector<uint16_t> *>>> mel_ops;

static const double hz_split = 1000.0;
static const double mel_split = 3 * 1000 / 200.0;
static const double mel_logstep = log(6.4) / 27.0;

static double hz_to_mel(double hz) {
  if (hz < hz_split)
    return 3 * hz / 200.0;
  else
    return mel_split + log(hz / hz_split) / mel_logstep;
}

static double mel_to_hz(double mel) {
  if (mel < mel_split)
    return 200 * mel / 3.0;
  else
    return 1000 * exp(mel_logstep * (mel - mel_split));
}

static const std::vector<uint16_t>& get_mel_ops(int N, int M, int lo, int hi) {
  auto entry = mel_ops.find(N);
  if (entry == mel_ops.end()) {
    mel_ops[N] = std::list<std::tuple<int,int,int,std::vector<uint16_t> *>>();
    entry = mel_ops.find(N);
  }
  for (auto it : entry->second) {
    if (std::get<0>(it) == M &&
	std::get<1>(it) == lo &&
	std::get<2>(it) == hi) {
      return *std::get<3>(it);
    }
  }

  const double fmin = (double)lo;
  const double fmax = (double)hi;

  double off = hz_to_mel(fmin);
  double rng = hz_to_mel(fmax) - off;

  // split frequency range lo to hi into N-1 bins
  uint16_t prev = 0;
  std::vector<uint16_t> *ops = new std::vector<uint16_t>();
  for (int i = 0; i < N; i++) {
    double hz = mel_to_hz(off + (i * rng)/(N-1));
    uint16_t pos = (uint16_t)floor((hz * M) / fmax);
    uint16_t cnt = pos - prev;
    if (i > 0) {
      double frac = 1.0 / cnt;
      uint16_t fx = (uint16_t)floor(frac * 128);
      ops->push_back((cnt << 8) + fx);
    } else {
      ops->push_back(cnt);
    }
    prev = pos;
  }

  entry->second.push_back(std::make_tuple(M, lo, hi, ops));

  return *ops;
}

void mel_spectrum_u16(int N, int M, int lo, int hi, uint16_t *spectrum) {
  uint16_t samp;
  uint16_t inst;
  uint8_t cnt;
  uint8_t frac;
  uint32_t prod;
  uint32_t acc;
  uint16_t mel;
  
  auto ops = get_mel_ops(N, M, lo, hi);
  
  int ip = ops[0];
  int op = 0;
  
  for (int i = 1; i < M; i++) {
    inst = ops[i];
    cnt = (uint8_t)(inst >> 8);
    frac = (uint8_t)inst;
    acc = 0;
    while (cnt) {
      prod = spectrum[ip++] * frac;
      acc = acc + prod;
      cnt--;
    }
    mel = (uint16_t)(acc >> 7);
    spectrum[op++] = mel;
  }
}
