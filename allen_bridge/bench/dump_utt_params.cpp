// dump_utt_params.cpp — host-side dumper for the extrapUTT chart parametrization.
//
// Loads the PRODUCTION KalmanParametrizations (the 19v0-style chart tables read
// from ParametrizedKalmanFit/25v0/params_UTT_v0.tab) with the verbatim Allen
// loader, then writes:
//   <out>/utt_struct.bin : raw sizeof(KalmanParametrizations) bytes of the
//                          loaded struct (POD; uploaded verbatim to the device
//                          in the micro-bench and reinterpreted by an
//                          ABI-identical struct compiled with NVRTC).
//   <out>/utt_meta.bin   : 19 float32 dev_UTT_META values, in the order the
//                          device extrapUTT() documents (00 ZINI ... 18 DEGY2).
//   <out>/utt_info.txt   : sizeof + key scalars, for the Python self-test.
//
// Allen stays READ-ONLY: we #include its headers, never edit them. This mirrors
// exactly what Allen does in production — KalmanParametrizations is memcpy'd to
// the device as constants.dev_kalman_params.
//
// Build: see build_bench_host.sh (g++, CPU only, no GPU needed).

#include "cuda_compat.h"  // CPU shim: defines __device__/__host__ as no-ops

#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "KalmanParametrizations.cuh"  // verbatim Allen header

using namespace ParKalmanFilter;

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " <param_data_dir> <out_dir>\n";
    return 1;
  }
  const std::string param_dir = argv[1];
  const std::string out_dir = argv[2];

  auto* params = new KalmanParametrizations();
  params->SetParameters(param_dir);
  if (!params->paramsLoaded) {
    std::cerr << "ERROR: parametrization not loaded from " << param_dir << "\n";
    return 2;
  }

  // dev_UTT_META in the order documented inside device extrapUTT().
  float META[19] = {
    params->ZINI, params->ZFIN, params->PMIN, params->BENDX,
    params->BENDX_X2, params->BENDX_Y2, params->BENDY_XY, params->Txmax,
    params->Tymax, params->XFmax, params->Dtxy, (float) params->Nbinx,
    (float) params->Nbiny, (float) params->XGridOption, (float) params->YGridOption,
    (float) params->DEGX1, (float) params->DEGX2, (float) params->DEGY1, (float) params->DEGY2};

  const std::string s_struct = out_dir + "/utt_struct.bin";
  const std::string s_meta = out_dir + "/utt_meta.bin";
  const std::string s_info = out_dir + "/utt_info.txt";

  {
    std::ofstream f(s_struct, std::ios::binary);
    f.write(reinterpret_cast<const char*>(params), sizeof(KalmanParametrizations));
  }
  {
    std::ofstream f(s_meta, std::ios::binary);
    f.write(reinterpret_cast<const char*>(META), sizeof(META));
  }
  {
    std::ofstream f(s_info);
    f << "sizeof_KalmanParametrizations " << sizeof(KalmanParametrizations) << "\n";
    f << "ZINI " << params->ZINI << "\n";
    f << "ZFIN " << params->ZFIN << "\n";
    f << "PMIN " << params->PMIN << "\n";
    f << "Nbinx " << params->Nbinx << "\n";
    f << "Nbiny " << params->Nbiny << "\n";
    f << "DEGX1 " << params->DEGX1 << " DEGX2 " << params->DEGX2
      << " DEGY1 " << params->DEGY1 << " DEGY2 " << params->DEGY2 << "\n";
    // offsets of a few members, so the device port can be cross-checked
    f << "off_ZINI " << (size_t)((char*) &params->ZINI - (char*) params) << "\n";
    f << "off_Nbinx " << (size_t)((char*) &params->Nbinx - (char*) params) << "\n";
    f << "off_paramsLoaded " << (size_t)((char*) &params->paramsLoaded - (char*) params) << "\n";
  }

  std::printf(
    "dumped: sizeof=%zu  ZINI=%.1f ZFIN=%.1f PMIN=%.1f Nbin=%dx%d DEG=(%d,%d,%d,%d)\n",
    sizeof(KalmanParametrizations), params->ZINI, params->ZFIN, params->PMIN, params->Nbinx,
    params->Nbiny, params->DEGX1, params->DEGX2, params->DEGY1, params->DEGY2);
  std::printf("wrote %s (%zu B), %s (76 B), %s\n", s_struct.c_str(), sizeof(KalmanParametrizations),
              s_meta.c_str(), s_info.c_str());
  delete params;
  return 0;
}
