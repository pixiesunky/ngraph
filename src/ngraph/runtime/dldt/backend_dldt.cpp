//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#if defined(NGRAPH_TBB_ENABLE)
#include <tbb/tbb_stddef.h>
#endif

#include "dldt_backend_visibility.hpp"

#include "ngraph/component_manager.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/dldt/backend_dldt.hpp"
#include "ngraph/runtime/dldt/dldt_tensor_view.hpp"
#include "ngraph/util.hpp"

#ifdef NGRAPH_MLIR_ENABLE
#include "contrib/mlir/backend/cpu/cpu_backend.hpp"
#include "contrib/mlir/core/compiler.hpp"
#endif

using namespace ngraph;
using namespace std;

extern "C" DLDT_BACKEND_API void ngraph_register_dldt_backend()
{
    runtime::BackendManager::register_backend("DLDT", [](const std::string& /* config */) {
        static bool is_initialized = false;
        if (!is_initialized)
        {
#if defined(NGRAPH_TBB_ENABLE)
            // Force TBB to link to the backend
            tbb::TBB_runtime_interface_version();
#endif
            ngraph::runtime::cpu::register_builders();
            is_initialized = true;
        }
        return make_shared<runtime::cpu::DLDT_Backend>();
    });
}
