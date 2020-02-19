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

//#pragma once

#include <ie_core.hpp>
#include <string>
#include "ngraph/opsets/opset.hpp"
#include "ngraph/runtime/dldt/dldt_tensor_view.hpp"
#include "ngraph/runtime/tensor.hpp"
//#include "ngraph/ngraph.hpp"

InferenceEngine::Blob::Ptr fill_blob(InferenceEngine::SizeVector shape, std::vector<float> data);

class Handle;

namespace ngraph
{
    namespace runtime
    {
        namespace dldt
        {
            class DLDT_Executable : public Executable
            {
		InferenceEngine::CNNNetwork network;
        	std::string device;
            public:
                DLDT_Executable(std::shared_ptr<Function> func, std::string _device)
                {
                    auto opset = ngraph::get_opset1();
                    bool all_opset1 = true;
                    for (auto& node : func->get_ops())
                    {
                        if (!opset.contains_op_type(node.get()))
                        {
                            all_opset1 = false;
                            break;
                        }
                    }

                    if (!all_opset1)
                    {
                        std::cout << "UNSUPPORTED OPS DETECTED!" << std::endl;
                        THROW_IE_EXCEPTION << "Exit from test";
                    }
                    std::cout << "Nodes in test: ";
                    for (auto& node : func->get_ops())
                    {
                        std::cout << node->get_type_info().name << " ";
                    }
                    std::cout << std::endl;
                    network = InferenceEngine::CNNNetwork(func);
                    device = _device;
                }
                ~DLDT_Executable() = default;

                bool call(const std::vector<std::shared_ptr<dldt::DLDTTensorView>>& outputs,
                          const std::vector<std::shared_ptr<dldt::DLDTTensorView>>& inputs)
                {
                    try
                    {
                        InferenceEngine::Core ie;

                        //  Loading model to the plugin (BACKEND_NAME)
                        InferenceEngine::ExecutableNetwork exeNetwork =
                            ie.LoadNetwork(network, device);
                        //  Create infer request
                        InferenceEngine::InferRequest inferRequest =
                            exeNetwork.CreateInferRequest();
                        //  Prepare input and output blobs
                        InferenceEngine::InputsDataMap inputInfo = network.getInputsInfo();

                        if (inputInfo.size() != inputs.size())
                        {
                            THROW_IE_EXCEPTION
                                << "Function inputs number differ from number of given inputs";
                        }

                        size_t i = 0;
                        for (auto& it : inputInfo)
                        {
                            size_t size = inputs[i]->data.size() / sizeof(float);
                            float* orig_data = (float*)inputs[i]->data.data();
                            std::vector<float> data(orig_data, orig_data + size);
                            inferRequest.SetBlob(
                                it.first, fill_blob(it.second->getTensorDesc().getDims(), data));
                            i++;
                        }

                        //  Prepare output blobs
                        std::string output_name = network.getOutputsInfo().begin()->first;

                        inferRequest.Infer();
                        InferenceEngine::Blob::Ptr output = inferRequest.GetBlob(output_name);

                        float* output_ptr = output->buffer().as<float*>();
                        // TODO: how to get size without explicit calculation?
                        size_t size = 1;
                        for (const auto& dim : output->getTensorDesc().getDims())
                        {
                            size *= dim;
                        }
                        //  Vector initialization from pointer
                        std::vector<float> result(output_ptr, output_ptr + size);
                        outputs[0]->write(result.data(), result.size() * sizeof(float));
                        return true;
                    }
                    catch (...)
                    {
                        THROW_IE_EXCEPTION << "FAILED";
                    }
                }
            };
	    class DLDT_Backend : public runtime::Backend
	    {
		public:
		std::string device; 
		DLDT_Backend(const std::string & _device) : device(_device) { }
		~DLDT_Backend() { }
		static std::shared_ptr<Backend> create(const std::string device,
                                                                  bool must_support_dynamic = false) {
			return std::shared_ptr<dldt::DLDT_Backend>(new dldt::DLDT_Backend(device));
                }
		std::shared_ptr<ngraph::runtime::Tensor> create_tensor(const ngraph::element::Type& element_type, const Shape& shape) {
			return std::make_shared<dldt::DLDTTensorView>(element_type, shape);
		}
		std::shared_ptr<ngraph::runtime::Tensor> create_tensor(const ngraph::element::Type& element_type, const Shape& shape, void* memory_pointer) {
			return std::make_shared<dldt::DLDTTensorView>(element_type, shape, memory_pointer);
		}

		std::shared_ptr<Executable> compile(std::shared_ptr<Function> func, bool enable_performance_data = false) {
			return std::make_shared<DLDT_Executable>(func, device);
		}	
		bool is_supported(const Node& node) const {
			return false;
		}
		bool is_supported_property(const Property prop) const;
		//void remove_compiled_function(std::shared_ptr<DLDT_Executable> exec);
	    };
		/*
            class DLDT_Backend : public runtime::Backend
            {
            public:
                static std::shared_ptr<dldt::DLDT_Backend> create(const std::string device,
                                                                  bool must_support_dynamic = false)
                {
                    return std::shared_ptr<Backend>(new Backend(device));
                }

                DLDT_Backend(std::string _device)
                    : device(_device)
                {
                }

                std::shared_ptr<dldt::DLDTTensorView> create_tensor(ngraph::element::Type type,
                                                                    ngraph::Shape shape)
                {
                    return std::make_shared<dldt::DLDTTensorView>(type, shape);
                }

                template <typename T>
                std::shared_ptr<dldt::DLDTTensorView>
                    create_tensor(ngraph::element::Type type, ngraph::Shape shape, T* data)
                {
                    auto tensor = std::make_shared<dldt::DLDTTensorView>(type, shape);
                    size_t size = 1;
                    for (auto x : shape)
                    {
                        size *= x;
                    }
                    std::vector<T> v(data, data + size);
                    tensor->write(data, size * sizeof(T));
                    return tensor;
                }

                template <class T>
                std::shared_ptr<dldt::DLDTTensorView> create_tensor(ngraph::Shape shape)
                {
                    return std::make_shared<dldt::DLDTTensorView>(ngraph::element::from<T>(),
                                                                  shape);
                }

                std::shared_ptr<dldt::DLDTTensorView>
                    create_dynamic_tensor(ngraph::element::Type type, ngraph::PartialShape shape)
                {
                    return std::make_shared<dldt::DLDTTensorView>(type, shape);
                }

                bool supports_dynamic_tensors() { return true; }
                std::shared_ptr<Executable> compile(std::shared_ptr<Function> func)
                {
                    return std::make_shared<Executable>(func, device);
                }

                std::string device;
            };*/
            };
        }
    }
}
