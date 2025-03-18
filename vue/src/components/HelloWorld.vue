<template>
  <div class="hello">
    <div class="left-panel">
      <ImageUploader />
      <ResultDisplay />
    </div>
    <div class="right-panel">
      <el-button type="primary">按钮 1</el-button>
      <el-button type="success">按钮 2</el-button>
      <el-button type="warning">按钮 3</el-button>
      <el-button type="info">按钮 4</el-button>
      <el-button type="danger">按钮 5</el-button>
    </div>
  </div>
</template>

<script>
import ImageUploader from './ImageUploader.vue';
import ResultDisplay from './ResultDisplay.vue';

export default {
  name: 'HelloWorld',
  components: {
    ImageUploader,
    ResultDisplay
  }
}
</script>

<style scoped>
.hello {
  display: flex;
}

.left-panel {
  flex: 1;
  padding: 20px;
}

.right-panel {
  display: flex;
  flex-direction: column;
  justify-content: space-around;
  padding: 20px;
}
</style>