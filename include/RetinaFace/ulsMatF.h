//
// Created by rgd on 19-6-10.
//

#ifndef RETINAFACE_NCNN_ULSMATF_H
#define RETINAFACE_NCNN_ULSMATF_H


class ulsMatF{
public:
    float* m_data;
    int m_rows, m_cols, m_channels;

public:
    ulsMatF(int cols, int rows, int channels){
        m_rows = rows;
        m_cols = cols;
        m_channels = channels;
        int size = channels * rows * cols * sizeof(float);
        m_data = (float*)malloc(size);
        memset((void *)m_data, 0, size);
    }
    ~ulsMatF(){
        if(m_data) free(m_data);
    }

    float *at(int channel, int row, int col){
        assert(m_data != NULL);
        assert(row < m_rows);
        assert(col < m_cols);
        assert(channel < m_channels);

        return m_data + (channel * m_rows * m_cols) + row * m_cols + col;
    }

    int getRows() {return m_rows;}
    int getCols() {return m_cols;}
    int getChannels() {return m_channels;}
};

#endif //RETINAFACE_NCNN_ULSMATF_H
