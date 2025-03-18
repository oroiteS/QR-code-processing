<template>
  <div class="image-uploader">
    <h3>上传图片</h3>
    <el-upload
      class="upload-container"
      action="#"
      :auto-upload="false"
      :on-change="handleChange"
      :show-file-list="true"
      :limit="1"
      accept="image/*"
      drag>
      <template #default>
        <el-icon class="el-icon--upload"><Upload /></el-icon>
        <div class="el-upload__text">将图片拖到此处，或<em>点击上传</em></div>
      </template>
      <template #tip>
        <div class="el-upload__tip">只能上传 JPG/PNG 格式图片文件</div>
      </template>
    </el-upload>
    <div v-if="imageUrl" class="preview">
      <el-divider content-position="left">图片预览</el-divider>
      <div class="preview-container">
        <img :src="imageUrl" class="preview-image" />
      </div>
      <el-button type="primary" @click="processImage" class="mt-3">
        <el-icon><Check /></el-icon>
        确认使用此图片
      </el-button>
    </div>
  </div>
</template>

<script>
import { Upload, Check } from '@element-plus/icons-vue'

export default {
  name: 'ImageUploader',
  components: {
    Upload,
    Check
  },
  data() {
    return {
      imageUrl: '',
      imageFile: null
    }
  },
  methods: {
    handleChange(uploadFile) {
      this.imageFile = uploadFile.raw;
      this.imageUrl = URL.createObjectURL(uploadFile.raw);
    },
    processImage() {
      this.$emit('process-image', this.imageFile);
    }
  }
}
</script>

<style scoped>
.image-uploader {
  margin-bottom: 20px;
}

.upload-container {
  width: 100%;
  border: 1px dashed #d9d9d9;
  border-radius: 6px;
  cursor: pointer;
  position: relative;
  overflow: hidden;
  transition: border-color 0.3s;
}

.upload-container:hover {
  border-color: #409eff;
}

.el-icon--upload {
  font-size: 28px;
  color: #8c939d;
  margin: 16px 0 16px;
  line-height: 50px;
}

.preview {
  margin-top: 20px;
}

.preview-container {
  display: flex;
  justify-content: center;
  align-items: center;
  margin: 15px 0;
  background-color: #f5f7fa;
  border-radius: 4px;
  padding: 10px;
}

.preview-image {
  max-width: 100%;
  max-height: 200px;
  object-fit: contain;
  border-radius: 4px;
}

.mt-3 {
  margin-top: 15px;
  width: 100%;
}
</style>