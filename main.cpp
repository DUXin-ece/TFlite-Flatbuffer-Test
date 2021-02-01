#include <iostream>
#include <fstream>
#include "schema_generated.h"
#include <string>

using namespace std;
using namespace tflite;

const char *filename = "C:\\Users\\Xin Du\\Desktop\\TFliteTest\\TFliteTest\\model.tflite";

const vector<string> TensorType_String = {
  "FLOAT32",
  "FLOAT16 ",
  "INT32 ",
  "UINT8 ",
  "INT64 ",
  "STRING ",
  "BOOL ",
  "INT16 ",
  "COMPLEX64 ",
  "INT8 ",
  "FLOAT64 ",
  "COMPLEX128 ",
  "UINT64 ",
  "RESOURCE ",
  "VARIANT ",
};
const vector<string> Operator_String = {
"ADD" ,
"AVERAGE_POOL_2D",
"CONCATENATION",
"CONV_2D",
"DEPTHWISE_CONV_2D",
"DEPTH_TO_SPACE",
"DEQUANTIZE",
"EMBEDDING_LOOKUP",
"FLOOR",
"FULLY_CONNECTED",
"HASHTABLE_LOOKUP",
"L2_NORMALIZATION",
"L2_POOL_2D",
"LOCAL_RESPONSE_NORMALIZATION",
"LOGISTIC",
"LSH_PROJECTION",
"LSTM",
"MAX_POOL_2D",
"MUL",
"RELU",
"RELU_N1_TO_1 = 20",
"RELU6",
"RESHAPE",
"RESIZE_BILINEAR",
"RNN",
"SOFTMAX",
"SPACE_TO_DEPTH",
"SVDF",
"TANH",
"CONCAT_EMBEDDINGS",
"SKIP_GRAM",
"CALL",
"CUSTOM",
"EMBEDDING_LOOKUP_SPARSE",
"PAD",
"UNIDIRECTIONAL_SEQUENCE_RNN",
"GATHER",
"BATCH_TO_SPACE_ND",
"SPACE_TO_BATCH_ND",
"TRANSPOSE",
"MEAN",
"SUB",
"DIV",
"SQUEEZE",
"UNIDIRECTIONAL_SEQUENCE_LSTM",
"STRIDED_SLICE",
"BIDIRECTIONAL_SEQUENCE_RNN",
"EXP",
"TOPK_V2",
"SPLIT",
"LOG_SOFTMAX",
"DELEGATE",
"BIDIRECTIONAL_SEQUENCE_LSTM",
"CAST",
"PRELU",
"MAXIMUM",
"ARG_MAX",
"MINIMUM",
"LESS",
"NEG",
"PADV2",
"GREATER",
"GREATER_EQUAL",
"LESS_EQUAL",
"SELECT",
"SLICE",
"SIN",
"TRANSPOSE_CONV",
"SPARSE_TO_DENSE",
"TILE",
"EXPAND_DIMS",
"EQUAL",
"NOT_EQUAL",
"LOG",
"SUM",
"SQRT",
"RSQRT",
"SHAPE",
"POW",
"ARG_MIN",
"FAKE_QUANT",
"REDUCE_PROD ",
"REDUCE_MAX",
"PACK",
"LOGICAL_OR",
"ONE_HOT",
"LOGICAL_AND",
"LOGICAL_NOT",
"UNPACK",
"REDUCE_MIN",
"FLOOR_DIV",
"REDUCE_ANY",
"SQUARE",
"ZEROS_LIKE",
"FILL",
"FLOOR_MOD",
"RANGE",
"RESIZE_NEAREST_NEIGHBOR",
"LEAKY_RELU",
"SQUARED_DIFFERENCE",
"MIRROR_PAD",
"ABS = 101",
"SPLIT_V",
"UNIQUE",
"CEIL",
"REVERSE_V2",
"ADD_N",
"GATHER_ND",
"COS",
"WHERE",
"RANK",
"ELU",
"REVERSE_SEQUENCE",
"MATRIX_DIAG",
"QUANTIZE",
"MATRIX_SET_DIAG",
"ROUND",
"HARD_SWISH",
"IF",
"WHILE",
"NON_MAX_SUPPRESSION_V4",
"NON_MAX_SUPPRESSION_V5",
"SCATTER_ND",
"SELECT_V2",
"DENSIFY",
"SEGMENT_SUM",
"BATCH_MATMUL",
"PLACEHOLDER_FOR_GREATER_OP_CODES",
"CUMSUM = 128",
"CALL_ONCE",
"BROADCAST_TO",
"RFFT2D",
"CONV_3D",

};

int main(void){
    long start,end;
    long size;
    unsigned char *model;

    /*****************Load Tflite model*******************/
    
    ifstream inFile(filename,ios_base::in|ios_base::binary);
    if (!inFile.good())
        throw runtime_error("Cannot open file");
    start = inFile.tellg();   //in fact l equals to 0
    inFile.seekg(0, ios::end);
    end = inFile.tellg();
    size = end-start;
    cout << "size of " << filename;
    cout << " is " << size << " bytes.\n";

    inFile.seekg(0,ios::beg);
    model = new unsigned char[size];
    inFile.read((char *)model,size);
    inFile.clear();
    inFile.close();
    /***************Read Tflite model data****************/

    auto tfmodel = GetModel(model);
    //Get version and description information
    auto version = tfmodel->version();
    cout<<"tflite version: "<<version<<endl;
    string description = tfmodel->description()->str();
    cout<<"description: "<<description<<endl;
    cout<<endl;

    //Read all operators used in this model
    //All operators used in this model are saved in operator_codes.
    auto ops_types = tfmodel->operator_codes()->size();
    cout<<ops_types<<endl; //The hello_world demo only has one operator - fully connected(== 9)
    int ops_index;
    for(int i=0;i<ops_types;i++){
        ops_index = tfmodel->operator_codes()->Get(i)->builtin_code();
        cout<<"Operator code "<<i<<": "<<ops_index<<"\t"<<Operator_String[ops_index]<<endl;
    }
    cout<<endl;
    
    //Get subgraph infomation
    auto subgraph_size = tfmodel->subgraphs()->size();
    cout<<"The total number of subgraphs: "<<subgraph_size<<endl;

    for(int i=0;i<subgraph_size;i++){
        auto subgraph_name = tfmodel->subgraphs()->Get(i)->name()->str();
        auto subgraph_ops = tfmodel->subgraphs()->Get(i)->operators()->size();
        cout<<"Subgraph "<<i<<": "<<subgraph_name<<endl;
        for(int j=0;j<subgraph_ops;j++){
            auto subgraph_ops = tfmodel->subgraphs()->Get(i)->operators()->Get(j);
            auto subgraph_ops_index = subgraph_ops->opcode_index();
            cout<<j<<":\t"<<subgraph_ops_index<<"\t";
            auto temp = tfmodel->operator_codes()->Get(subgraph_ops_index)->builtin_code();
            cout << Operator_String[temp] << endl;
            cout << "\t" << "input tensors: ";
            for (int k = 0; k < subgraph_ops->inputs()->size(); k++) {
                cout << subgraph_ops->inputs()->Get(k) << "\t";
            }
            cout << endl;
            cout << "\t" << "output tensors: ";
            for (int k = 0; k < subgraph_ops->outputs()->size(); k++) {
                cout << subgraph_ops->outputs()->Get(k) << "\t";
            }
            cout << endl;
        }
    }
    cout<<endl;

    //Get tensors in subgraph
    for(int i=0;i<subgraph_size;i++){
        auto tensor_num = tfmodel->subgraphs()->Get(i)->tensors()->size();
        cout<<"Tensors in subgraph "<<i<<": "<<tensor_num<<endl;
        for(int j=0;j<tensor_num;j++){
            auto tensor = tfmodel->subgraphs()->Get(i)->tensors()->Get(j);
            cout<<"name: "<<tensor->name()->str()<<"\t";
            cout << "type: " << TensorType_String[tensor->type()] << "\t";
            cout<<"shape: ";
            for(int k=0;k<tensor->shape()->size();k++){
                cout<<tensor->shape()->Get(k)<<" ";
            }
            cout<<"\t";
            cout<<"buffer index: "<<tensor->buffer();
            cout<<endl;

            auto buffer_i = tfmodel->buffers()->Get(tensor->buffer());
             if(buffer_i->data()){
                 for(int j=0;j<buffer_i->data()->size();j++){
                     cout<<(int)buffer_i->data()->Get(j)<<"\t";
                     if(j%10==9) cout<<endl;
                 }
                 cout<<endl;
             }

            cout<<endl;

        }
        cout<<endl;
    }
    



    

    /************Delete the pointer to tflite model*******/
    delete [] model;

    return 0;
}

