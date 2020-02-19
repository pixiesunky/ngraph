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

#include "ngraph/runtime/dldt/dldt_backend_visibility.hpp"

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

#include <ie_core.hpp>
//#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;
using namespace InferenceEngine;

Blob::Ptr fill_blob(SizeVector shape, std::vector<float> data)
{
    Layout layout;
    switch (shape.size())
    {
    case 1: layout = Layout::C; break;
    case 2: layout = Layout::NC; break;
    case 3: layout = Layout::CHW; break;
    case 4: layout = Layout::NCHW; break;
    case 5: layout = Layout::NCDHW; break;
    default: THROW_IE_EXCEPTION << "Can't convert dims " << shape.size() << " to Layout!";
    }
    MemoryBlob::Ptr blob(new TBlob<float>({Precision::FP32, shape, layout}));
    blob->allocate();
    float* blob_ptr = blob->rwmap().as<float*>();
    for (int i = 0; i < data.size(); i++)
    {
        blob_ptr[i] = data[i];
    }
    return blob;
}

extern "C" DLDT_BACKEND_API void ngraph_register_dldt_backend()
{
    ngraph::runtime::BackendManager::register_backend("DLDLT", [](const std::string& config) {
        return std::make_shared<ngraph::runtime::dldt::DLDT_Backend>(config);
    });
}
