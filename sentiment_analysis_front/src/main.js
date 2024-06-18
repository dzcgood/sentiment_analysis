import { createApp } from 'vue'
import App from './App.vue'
import router from './router/index.js'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import zhCn from 'element-plus/dist/locale/zh-cn.mjs'
import * as ElementPlusIconsVue from '@element-plus/icons-vue' // element-plus icon

//解决跨域
import axios from "axios";

const app = createApp(App)



// 挂载路由
app.use(router)

// emelemt-plus
app.use(ElementPlus, {
    locale: zhCn,
  })

app.config.globalProperties.$http = axios;

// 注册icon
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
    app.component(key, component)
}

app.mount('#app')
