#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include "model.h" 

namespace py = pybind11;

// 
PYBIND11_MODULE(llm_engine, m) {
    m.doc() = "C++ LLM Inference Engine Backend"; 
  
    py::class_<Transformer>(m, "Transformer")
        // 绑定构造函数：对应 Transformer(std::string &model_path)
        .def(py::init<const char*>()) 
        // 绑定前向传播函数：对应 vector<float> forward(int token, int pos)
        .def("forward", &Transformer::forward);
}