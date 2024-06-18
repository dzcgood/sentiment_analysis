<template>
  <div>
    <el-container class="container-layout">
      <el-header>
        <div>
          <el-icon :size="23"><Histogram /></el-icon>
          <span>情感分析系统</span>
        </div>
        <el-button type="info" @click="exit">退出</el-button>
      </el-header>
      <el-container>
        <!-- aside -->
        <el-aside width="200px">
          <el-menu :default-active="cur_index" @select="selectMenu">
            <div v-for="(item, index) in menu" :key="index">
              <router-link :to="{ path: item.router }">
                <el-menu-item :index="item.id">
                  <el-icon>
                    <!--这里引用了注册的组件-->
                    <component :is="item.icon"></component>
                  </el-icon>
                  <span>{{ item.title }}</span>
                </el-menu-item>
              </router-link>
            </div>
          </el-menu>
        </el-aside>
        <!-- main -->
        <el-main>
          <router-view></router-view>
        </el-main>
        <!-- footer -->
      </el-container>
      <!-- <el-footer>Footer</el-footer> -->
    </el-container>
  </div>
</template>

<script>
import { reactive, ref, shallowRef, onMounted } from "vue";
import { Document, Histogram, User, Search } from "@element-plus/icons-vue";

export default {
  components: {
    Document,
    Histogram,
    User,
    Search,
  },
  setup() {
    const menuList = [
      {
        id: "1",
        icon: shallowRef(Document),
        title: "文本情感分析",
        router: "sentiment_analysis",
        subMenu: [], // 是否有二级菜单
      },
      {
        id: "2",
        icon: shallowRef(Search),
        title: "微博情感分析",
        router: "analysis_by_weibo_id",
        subMenu: [], // 是否有二级菜单
      },
      {
        id: "3",
        icon: shallowRef(User),
        title: "微博用户情感分析",
        router: "analysis_weibo_user",
        subMenu: [], // 是否有二级菜单
      },
    ];
    const menu = reactive(menuList);

    var cur_index = ref("1");
    // 菜单激活回调
    function selectMenu(index, indexPath) {
      // 把当前选中的menuId存起来
      localStorage.setItem("menuId", JSON.stringify(index));
    }
    // 恢复之前选择的menuId界面
    onMounted(() => {
      var index = JSON.parse(localStorage.getItem("menuId"));
      if(index != null) {
        cur_index.value = index
      }
      
    });

    return { menu, cur_index, selectMenu };
  },
};
</script>

<style>
.container-layout {
  height: 100%;
  width: 70%;
  margin: 0 15%;
  position: fixed;
}

.container-layout .el-header {
  text-align: left;
  position: relative;
  border-bottom: 1px solid var(--el-border-color);
  color: var(--el-text-color-primary);
}

.container-layout .el-main {
  padding: 0;
  text-align: center;
  /* vertical-align: middle; */
}

.container-layout .el-footer {
  text-align: center;
  position: relative;
  background-color: var(--el-color-primary-light-7);
  color: var(--el-text-color-primary);
}

.el-menu {
  align-items: center; /* 垂直居中 */
  justify-content: center; /* 水平居中 */
}

.el-header {

  display: flex;

  justify-content: space-between;

  align-items: center;

  color: #fff;

  font-size: 20px;
}

.el-header div {
  display: flex;

  align-items: center;
}

.el-header div span {
  margin-left: 15px;
}

.el-menu {
  height:100%
}
</style>
