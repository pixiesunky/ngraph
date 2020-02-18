# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

set(SRC
	backend_dldt.cpp
	dldt_tensor_view.cpp 
)

add_library(libdldt_backend SHARED ${SRC})
add_definitions(-DNGRAPH_DLDT_ENABLE)
target_include_directories(libdldt_backend PUBLIC ${IE_MAIN_SOURCE_DIR}/include)
target_link_libraries(libdldt_backend PRIVATE inference_engine)
target_compile_definitions(libdldt_backend PRIVATE DLDT_BACKEND_EXPORTS)

