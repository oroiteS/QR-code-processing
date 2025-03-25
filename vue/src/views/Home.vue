<template>
  <div class="home">
    <el-row :gutter="20">
      <el-col :span="16">
        <el-card shadow="hover" class="left-panel">
          <ImageUploader @process-image="handleProcessImage" />
          <ResultDisplay :result="result" :result-type="resultType" />
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card shadow="hover" class="right-panel">
          <h3>操作面板</h3>
          <el-divider></el-divider>
          <el-button type="primary" icon="Search" @click="handleDetect" :loading="loading">识别图像</el-button>
          <el-button type="success" icon="Position" @click="handleCorrect" :loading="loading">校正图像</el-button>
          <el-button type="warning" icon="Edit" @click="handleDeblur" :loading="loading">修复图像</el-button>
          <el-button type="danger" icon="Aim" @click="handleScan" :loading="loading">定位图像</el-button>
          <el-button type="info" icon="RefreshRight" @click="handleReset" :loading="loading">重置图像</el-button>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script>
import ImageUploader from '@/components/ImageUploader.vue';
import ResultDisplay from '@/components/ResultDisplay.vue';
import { ElMessage } from 'element-plus';

export default {
  name: 'HomeView',
  components: {
    ImageUploader,
    ResultDisplay
  },
  data() {
    return {
      result: null,
      resultType: 'text',
      currentImage: null,
      loading: false,
      apiBaseUrl: 'http://127.0.0.1:5000' // Flask后端API基础URL
    }
  },
  methods: {
    handleProcessImage(file) {
      this.currentImage = file;
      // 显示上传提示
      this.result = '图片已上传，请选择右侧操作按钮进行处理...';
      this.resultType = 'text';
    },
    
    async sendImageToApi(endpoint) {
      if (!this.currentImage) {
        ElMessage.warning('请先上传图片');
        return null;
      }

      this.loading = true;
      this.result = '处理中，请稍候...';
      this.resultType = 'text';

      try {
        const formData = new FormData();
        formData.append('image', this.currentImage);

        const response = await fetch(`${this.apiBaseUrl}/${endpoint}`, {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || `请求失败: ${response.status}`);
        }

        return await response;
      } catch (error) {
        console.error(`API请求错误:`, error);
        this.result = `处理失败: ${error.message}`;
        this.resultType = 'text';
        this.loading = false;
        return null;
      }
    },
    
    async handleDetect() {
      const response = await this.sendImageToApi('detect');
      if (!response) return;
      
      try {
        const data = await response.json();
        
        // 处理图像数据
        if (data.image) {
          const imageBytes = new Uint8Array(data.image.match(/.{1,2}/g).map(hex => parseInt(hex, 16)));
          const blob = new Blob([imageBytes], { type: 'image/jpeg' });
          this.result = URL.createObjectURL(blob);
          this.resultType = 'image';
          
          // 显示检测到的对象信息
          if (data.detected_objects && data.detected_objects.length > 0) {
            ElMessage.success(`成功检测到${data.detected_objects.length}个对象`);
          }
        }
      } catch (error) {
        this.result = `解析响应失败: ${error.message}`;
        this.resultType = 'text';
      } finally {
        this.loading = false;
      }
    },
    
    async handleCorrect() {
      const response = await this.sendImageToApi('correct');
      if (!response) return;
      
      try {
        // correct接口直接返回图片文件
        const blob = await response.blob();
        this.result = URL.createObjectURL(blob);
        this.resultType = 'image';
        ElMessage.success('图像校正成功');
      } catch (error) {
        this.result = `处理失败: ${error.message}`;
        this.resultType = 'text';
      } finally {
        this.loading = false;
      }
    },
    
    async handleDeblur() {
      const response = await this.sendImageToApi('deblur');
      if (!response) return;
      
      try {
        const data = await response.json();
        
        if (data.image) {
          const imageBytes = new Uint8Array(data.image.match(/.{1,2}/g).map(hex => parseInt(hex, 16)));
          const blob = new Blob([imageBytes], { type: 'image/jpeg' });
          this.result = URL.createObjectURL(blob);
          this.resultType = 'image';
          ElMessage.success('图像去模糊处理成功');
        }
      } catch (error) {
        this.result = `处理失败: ${error.message}`;
        this.resultType = 'text';
      } finally {
        this.loading = false;
      }
    },
    
    async handleScan() {
      const response = await this.sendImageToApi('scan');
      if (!response) return;
      
      try {
        const data = await response.json();
        this.result = data.message || '扫描完成';
        this.resultType = 'text';
        if (data.message.includes('成功')) {
          ElMessage.success('扫描成功');
        } else {
          ElMessage.warning('未检测到二维码内容');
        }
      } catch (error) {
        this.result = `处理失败: ${error.message}`;
        this.resultType = 'text';
      } finally {
        this.loading = false;
      }
    },
    
    handleReset() {
      if (!this.currentImage) {
        ElMessage.warning('没有可重置的图像');
        return;
      }
      
      this.result = URL.createObjectURL(this.currentImage);
      this.resultType = 'image';
      ElMessage.info('已重置为原始图像');
    }
  }
}
</script>

<style scoped>
.home {
  height: 100%;
}

.left-panel {
  height: 100%;
  padding: 20px;
}

.right-panel {
  height: 100%;
  padding: 20px;
  display: flex;
  flex-direction: column;
}

.right-panel .el-button {
  margin: 10px 0;
  height: 50px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.el-card {
  height: 100%;
}
</style>