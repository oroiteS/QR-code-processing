<template>
  <div class="result-display">
    <h3>处理结果</h3>
    <el-divider></el-divider>
    <el-empty v-if="!result" description="暂无处理结果" :image-size="150"></el-empty>
    <div v-else class="result-content">
      <div v-if="resultType === 'text'" class="text-result">
        <el-alert
          type="success"
          :closable="false">
          <div class="result-text">{{ result }}</div>
        </el-alert>
      </div>
      <div v-else-if="resultType === 'image'" class="image-result">
        <div class="image-container">
          <img :src="result" class="result-image" />
        </div>
        <div class="button-group">
          <el-button type="primary" size="large" @click="download" class="action-btn">
            <el-icon><Download /></el-icon>
            下载结果图片
          </el-button>
          <el-button type="success" size="large" @click="useAsInput" class="action-btn">
            <el-icon><Top /></el-icon>
            使用当前结果
          </el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { Download, Top } from '@element-plus/icons-vue'

export default {
  name: 'ResultDisplay',
  components: {
    Download,
    Top
  },
  props: {
    result: {
      type: [String, null],
      default: null
    },
    resultType: {
      type: String,
      default: 'text',
      validator: function(value) {
        return ['text', 'image'].includes(value);
      }
    }
  },
  methods: {
    download() {
      if (this.resultType === 'image' && this.result) {
        const link = document.createElement('a');
        link.href = this.result;
        link.download = `processed-image-${new Date().getTime()}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    },
    useAsInput() {
      if (this.resultType === 'image' && this.result) {
        this.$emit('use-as-input', this.result);
      }
    }
  }
}
</script>

<style scoped>
.result-display {
  margin-top: 30px;
  border-top: 1px solid #eaeaea;
  padding-top: 20px;
}

.result-content {
  margin-top: 20px;
  min-height: 200px;
}

.text-result {
  padding: 15px;
  border-radius: 4px;
  background-color: #f8f8f8;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.result-text {
  white-space: pre-line;
  line-height: 1.6;
}

.image-result {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.image-container {
  background-color: #f5f7fa;
  padding: 10px;
  border-radius: 4px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  margin-bottom: 15px;
}

.result-image {
  max-width: 100%;
  max-height: 300px;
  object-fit: contain;
}

.button-group {
  display: flex;
  justify-content: space-between;
  width: 100%;
  margin-top: 15px;
}

.action-btn {
  width: 48%;
}
</style>