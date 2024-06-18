<template>
  <div>
    <!--输入部分-->
    <div class="wrapper">
      <div class="shadow">
        <el-form :inline="true" :model="input">
          <div class="margin-bottom-0">
            <el-form-item label="微博ID" required>
              <el-input v-model="input.weiboId" placeholder="微博ID" />
            </el-form-item>
            <el-form-item label="cookie">
              <el-input v-model="input.cookie" placeholder="cookie" />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" @click="spiderGetSingleTweet"
                >爬虫一下</el-button
              >
            </el-form-item>
          </div>
        </el-form>
      </div>
    </div>
    <!-- https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png -->
    <!--展示部分-->
    <div class="wrapper" ref="loading">
      <div class="shadow">
        <!--用户信息-->
        <el-row align="bottom">
          <el-col :span="1" style="height: 50px">
            <el-avatar :size="50" :src="spideredTweet.avatarUrl" />
          </el-col>
          <el-col :span="23" id="user-info">
            <el-row>
              <span id="nickname">{{ spideredTweet.nickName }}</span>
            </el-row>
            <el-row id="desc">
              <span style="margin: 0 5px 0 0">{{
                spideredTweet.createdAt
              }}</span>
              <span>{{ spideredTweet.source }}</span>
            </el-row>
          </el-col>
        </el-row>
        <el-row style="margin: 20px 0">
          <el-col :span="22" :offset="1">
            <el-card
              shadow="hover"
              style="border: 2px dashed var(--el-border-color)"
            >
              <span>{{ spideredTweet.content }}</span>
            </el-card>
          </el-col>
        </el-row>

        <el-row justify="space-between" align="middle" id="do_analysis">
          <el-col :span="2" :offset="21">
            <el-button type="primary" @click="analysis">开始分析</el-button>
          </el-col>
        </el-row>
        <el-row align="middle" id="result">
          <el-col :span="3">情感分析结果</el-col>
          <el-col :span="1">
            <!--如果没有进行分析，就不展示-->
            <el-image
              v-if="spideredTweet.pred != -1"
              style="width: 50px; height: 50px"
              :src="imgUrlAndLabel[spideredTweet.pred].imgUrl"
              fit
            />
          </el-col>
          <!--如果没有进行分析，就不展示-->
          <el-col
            :span="2"
            :style="imgUrlAndLabel[spideredTweet.pred].style"
            v-if="spideredTweet.pred != -1"
            >{{ imgUrlAndLabel[spideredTweet.pred].label }}</el-col
          >
        </el-row>
      </div>
    </div>

    <!--历史部分-->
    <div id="history">
      <div class="wrapper">
        <div class="shadow">
          <el-row>
            <el-col :span="12">
              <span>爬取记录：</span>
            </el-col>
            <el-col :span="12" v-if="tableData.data.length == 0">
              <span>无相关记录</span>
            </el-col>
          </el-row>
        </div>
      </div>

      <div class="wrapper" v-if="tableData.data.length > 0">
        <div class="shadow">
          <el-row>
            <el-table
              :data="tableData.data"
              style="width: 100%"
              height="400"
              tableLayout="fixed"
              stripe
            >
              <el-table-column fixed prop="date" label="Date" />
              <el-table-column fixed prop="weiboId" label="微博ID" />
              <el-table-column prop="content" label="Content" />
              <el-table-column prop="result" label="Result" />
              <el-table-column
                fixed="right"
                label="Operation"
                width="120"
                type="index"
              >
                <template #default="scope">
                  <el-button
                    type="info"
                    icon="Delete"
                    circle
                    @click="removeOneHistory(scope.$index)"
                  />
                </template>
              </el-table-column>
            </el-table>
          </el-row>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, reactive, onMounted } from "vue";
import axios from "axios";
import { ElLoading, ElMessage, ElNotification } from "element-plus";

export default {
  setup() {
    const input = reactive({
      weiboId: "",
      cookie: "",
    });

    const loading = ref(null);

    // 防止微博403
    const imgBaseUrl = "https://image.baidu.com/search/down?url=";

    // 一条微博需要展示的信息
    const spideredTweet = reactive({
      weiboID: "",
      nickName: "人民日报",
      createdAt: "2023-03-13 12:32:53",
      avatarUrl:
        imgBaseUrl +
        "https://tvax4.sinaimg.cn/crop.0.0.1018.1018.1024/0033ImPzly8h8vgemh8kxj60sa0sadgw02.jpg?KID=imgbed,tva&Expires=1678727574&ssig=1nwOGFUYr4",
      source: "来自 微博网页版",
      content:
        "【这些话，振奋人心！】习近平在十四届全国人大一次会议上发表重要讲话。①#中国人民成为自己命运的主人#；②#坚定不移推进祖国统一进程#；③#中国的发展惠及世界#，中国的发展离不开世界。更多↓↓转发，学习！",
      pred: 0,
    });

    // 爬虫抓取微博信息
    function spiderGetSingleTweet() {
      if (input.weiboId.trim() == "") {
        ElMessage({
          message: "请输入要分析的微博ID",
          type: "warning",
          appendTo: loading.value,
          center: true,
        });
        return;
      }
      // spideredTweet.weiboID = input.weiboId
      const loadingIcon = ElLoading.service({
        // 一定得写loading.value 因为是ref
        target: loading.value,
        fullscreen: false,
        text: "Loading",
      });
      axios
        .post("/Api/single_tweet_analysis", {
          tweet_id: input.weiboId,
          cookie: input.cookie,
        })
        .then(function (response) {
          console.log(response);
          const data = response.data;
          spideredTweet.nickName = data.user.nick_name;
          spideredTweet.createdAt = data.created_at;
          spideredTweet.avatarUrl = data.user.avatar_hd;
          spideredTweet.source = "来自 " + data.source;
          spideredTweet.content = data.content;
          spideredTweet.pred = -1;
          // 关闭loading
          loadingIcon.close();
          // Message消息提示
          ElMessage({
            message: "爬虫完成",
            type: "success",
            appendTo: loading.value,
            center: true,
          });
        })
        .catch(function (error) {
          console.log(error);
        });
    }

    // 模型分析微博情感
    function analysis() {
      if (spideredTweet.content == "") {
        ElMessage({
          message: "请先获取微博信息",
          type: "warning",
          appendTo: loading.value,
          center: true,
        });
        return;
      }
      const loadingIcon = ElLoading.service({
        // 一定得写loading.value 因为是ref
        target: loading.value,
        fullscreen: false,
        text: "Loading",
      });
      axios
        .post("/Api/sentiment_analysis", {
          content: spideredTweet.content,
        })
        .then(function (response) {
          console.log(response);
          spideredTweet.pred = response.data["pred"];
          // 关闭loading
          loadingIcon.close();
          // Message消息提示
          ElMessage({
            message: "分析完成",
            type: "success",
            appendTo: loading.value,
            center: true,
          });
          setHistory(input.weiboId, spideredTweet.content, imgUrlAndLabel[spideredTweet.pred].label)
        })
        .catch(function (error) {
          console.log(error);
        });
    }

    const imgUrlAndLabel = reactive([
      {
        imgUrl: require("@/assets/happy.svg"),
        label: "积极",
        style: "color:#d81e06; text-align:center; font-weight:bold;",
      },
      {
        imgUrl: require("@/assets/mad.svg"),
        label: "愤怒",
        style: "color:#d81e06; text-align:center; font-weight:bold;",
      },
      {
        imgUrl: require("@/assets/sad.svg"),
        label: "悲伤",
        style: "color:#000000; text-align:center; font-weight:bold;",
      },
      {
        imgUrl: require("@/assets/fear.svg"),
        label: "恐惧",
        style: "color:#000000; text-align:center; font-weight:bold;",
      },
      {
        imgUrl: require("@/assets/surprised.svg"),
        label: "惊奇",
        style: "color:#75EAE4; text-align:center; font-weight:bold;",
      },
      {
        imgUrl: require("@/assets/neutral.svg"),
        label: "无情绪",
        style: "color:#000000; text-align:center; font-weight:bold;",
      },
    ]);
    
    // 历史记录
    const tableData = reactive({
      data: [
     
      ],
    });

    // 获取当前时间： 2023-03-17 13:21:00
    function getNowDate() {
      var myDate = new Date();
      var year = myDate.getFullYear(); //获取当前年
      var mon = myDate.getMonth() + 1; //获取当前月
      var date = myDate.getDate(); //获取当前日
      var hours = myDate.getHours(); //获取当前小时
      var minutes = myDate.getMinutes(); //获取当前分钟
      var seconds = myDate.getSeconds(); //获取当前秒
      var now =
        year +
        "-" +
        mon +
        "-" +
        date +
        " " +
        hours +
        ":" +
        minutes +
        ":" +
        seconds;
      return now;
    }

    // 添加一条历史记录
    function setHistory(weiboID, content, label) {
      console.log(tableData.data);
      if (tableData.data.length >= 20) {
        tableData.data.pop();
      }
      tableData.data.unshift({
        date: getNowDate(),
        weiboId: weiboID,
        content: content,
        result: label,
      });
      localStorage.setItem(
        "singleTweetAnalysisHistory",
        JSON.stringify(tableData.data)
      );
    }

    // 删除一条历史记录
    function removeOneHistory(index) {
      console.log(index);
      tableData.data.splice(index, 1);
      localStorage.setItem(
        "singleTweetAnalysisHistory",
        JSON.stringify(tableData.data)
      );
      ElNotification({
        title: "Success",
        message: "成功删除一条记录",
        type: "success",
      });
    }

    // 加载的时候读取历史记录
    onMounted(() => {
      var history = JSON.parse(
        localStorage.getItem("singleTweetAnalysisHistory")
      );
      if (history != null) {
        tableData.data = history;
      }
    });

    // RETURN
    return {
      input,
      loading,
      analysis,
      imgUrlAndLabel,
      tableData,
      spideredTweet,
      removeOneHistory,
      spiderGetSingleTweet,
    };
  },
};
</script>

<style scoped>
.wrapper {
  width: 90%;
  margin: 15px 5%;
}

.shadow {
  box-shadow: var(--el-box-shadow-light);
  width: 100%;
  padding: 10px;
  box-sizing: border-box;
  text-align: left;
}

.margin-bottom-0 .el-form-item {
  margin-bottom: 0 !important;
}

#user-info .el-row {
  padding-left: 15px;
}

#nickname {
  font-size: 14px;
  height: 25px;
  font-weight: bold;
}

#desc {
  font-size: 12px;
  color: #939393;
}

#do_analysis {
  border-bottom: 1px solid var(--el-border-color);
  padding-bottom: 10px;
}

#result {
  margin: 10px 0 0 0;
}
</style>
