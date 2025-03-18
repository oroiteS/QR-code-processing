<template>
  <div id="app">
    <el-container class="main-container">
      <el-header>
        <h1>图像处理应用</h1>
      </el-header>
      <el-main>
        <router-view />
      </el-main>
      <el-footer>
        <p>© 2025 图像处理应用</p>
      </el-footer>
    </el-container>
  </div>
</template>

<script>
export default {
  name: 'App'
  // 移除未使用的组件导入和注册
}
</script>

<style>
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: 'Helvetica Neue', Helvetica, 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', '微软雅黑', Arial, sans-serif;
  background-color: #f5f7fa;
  color: #333;
}

#app {
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.main-container {
  height: 100%;
}

.el-header {
  background-color: #409eff;
  color: white;
  display: flex;
  align-items: center;
  padding: 0 20px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.12);
}

.el-main {
  padding: 20px;
  height: calc(100vh - 120px);
  overflow-y: auto;
}

.el-footer {
  text-align: center;
  background-color: #f5f7fa;
  color: #909399;
  padding: 15px 0;
  border-top: 1px solid #e4e7ed;
}

.container {
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