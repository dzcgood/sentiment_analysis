<template>
  <div>
    <el-scrollbar height="900px">
      <!--输入部分-->
      <div class="wrapper" ref="userInfoLoading">
        <div class="shadow">
          <el-form :inline="true" :model="input">
            <div class="margin-bottom-0">
              <el-form-item label="微博用户ID" required>
                <el-input v-model="input.userId" placeholder="微博用户ID" />
              </el-form-item>
              <el-form-item label="cookie">
                <el-input v-model="input.cookie" placeholder="cookie" />
              </el-form-item>
              <el-form-item>
                <el-button type="primary" @click="spiderGetUser"
                  >爬虫一下</el-button
                >
              </el-form-item>
            </div>
          </el-form>
        </div>
      </div>

      <!--用户信息界面-->
      <div class="wrapper"  v-if="weiboItems.length != 0">
        <div class="shadow">
          <el-row>
            <el-col :span="12">
              <el-row align="bottom">
                <el-col :span="4" style="height: 100px">
                  <el-avatar :size="100" :src="weiboUser.avatar_hd" />
                </el-col>
                <el-col :span="20" id="user-info">
                  <el-row>
                    <span id="nickname">{{ weiboUser.nick_name }}</span>
                  </el-row>
                  <el-row id="desc">
                    <span style="margin: 0 5px 0 0"
                      >粉丝 {{ weiboUser.followers_count }}</span
                    >
                    <span>关注 {{ weiboUser.friends_count }}</span>
                  </el-row>
                </el-col>
              </el-row>
              <el-row style="padding: 20px 0 0 20px">
                <el-descriptions
                  class="margin-top"
                  :column="2"
                  size="default"
                  style="font-size: 14px"
                >
                  <el-descriptions-item :span="2">
                    <el-icon :size="15"><Document /></el-icon>
                    {{ weiboUser.description }}
                  </el-descriptions-item>
                  <el-descriptions-item>
                    <el-icon :size="15"><Clock /></el-icon>
                    {{ weiboUser.created_at }}
                  </el-descriptions-item>
                  <el-descriptions-item>
                    <el-icon :size="15"><Location /></el-icon>
                    {{ weiboUser.location }}
                  </el-descriptions-item>
                </el-descriptions>
              </el-row>
            </el-col>

            <el-col style="text-align: center" :span="12" v-if="resultId != -1">
              <el-row align="middle" style="margin-top: 20px">
                <el-col :span="8" :offset="3">
                  <el-image
                    style="width: 150px; height: 150px"
                    :src="imgUrlAndLabel[resultId].imgUrl"
                    fit
                  />
                </el-col>
                <el-col :span="2" :style="imgUrlAndLabel[resultId].style">
                  {{ imgUrlAndLabel[resultId].label }}
                </el-col>
              </el-row>
            </el-col>
          </el-row>
        </div>
      </div>

      <!--Echarts和词云-->
      <div class="wrapper" v-if="resultId != -1">
      <!-- <div class="wrapper"> -->
        <div class="shadow">
          <el-row style="text-align: center">
            <el-col :span="12">
              <div
                class="echart"
                id="myPieTable"
                ref="myPieTable"
                :style="myChartStyle"
              ></div>
            </el-col>
            <el-col :span="12">
              <!--词云-->
              <el-image
                style="width: 300px; height: 300px"
                :src="wordCloudUrl"
              />
            </el-col>
          </el-row>
        </div>
      </div>

      <div class="wrapper" v-if="weiboItems.length != 0">
        <div class="shadow">
          <el-row
            style="text-align: center; font-weight: bold"
            align="middle"
            justify="center"
          >
            <el-col :span="2"> 微博分析 </el-col>
            <el-col :span="2">
              <el-button type="primary" text @click="analysisTweets"
                >开始分析</el-button
              >
            </el-col>
          </el-row>
        </div>
      </div>

      <!--各条微博的情感分析-->

      <div
        v-for="(item, index) in weiboItems"
        :key="index"
        class="tweetWrapper"
        :ref="setItemRef"
      >
        <!--展示部分-->
        <div class="shadow">
          <!--用户信息-->
          <el-row align="bottom">
            <el-col :span="1" style="height: 50px">
              <el-avatar :size="50" :src="weiboUser.avatar_hd" />
            </el-col>
            <el-col :span="23" id="user-info">
              <el-row>
                <span id="nickname-small">{{ weiboUser.nick_name }}</span>
              </el-row>
              <el-row id="desc">
                <span style="margin: 0 5px 0 0">{{
                  weiboItems[index].created_at
                }}</span>
                <span>{{ weiboItems[index].ip_location }}</span>
              </el-row>
            </el-col>
          </el-row>
          <el-row style="margin: 20px 0">
            <el-col :span="22" :offset="1">
              <el-card
                shadow="hover"
                style="border: 2px dashed var(--el-border-color)"
              >
                <span>{{ weiboItems[index].content }}</span>
              </el-card>
            </el-col>
          </el-row>

          <el-row align="middle" id="result" v-if="predLabels[index] != -1">
            <el-col :span="3">情感分析结果</el-col>
            <el-col :span="1">
              <el-image
                style="width: 50px; height: 50px"
                :src="imgUrlAndLabel[predLabels[index]].imgUrl"
                fit
              />
            </el-col>
            <el-col :span="2" :style="imgUrlAndLabel[predLabels[index]].style">{{
              imgUrlAndLabel[predLabels[index]].label
            }}</el-col>
          </el-row>
        </div>
      </div>
    </el-scrollbar>
  </div>
</template>

<script>
import { ref, reactive, onUpdated} from "vue";
import * as echarts from "echarts";
import { ElLoading, ElMessage } from "element-plus";
import axios from "axios";

export default {
  setup() {
    const input = reactive({
      userId: "",
      cookie: "",
    });
    var urlTemp = ref("")
    const imgBaseUrl = "https://image.baidu.com/search/down?url=";
    var wordCloudUrl = ref("");

    const weiboUser = reactive({
      nick_name: "DzcGood",
      followers_count: 100,
      friends_count: 3,
      location: "北京 海淀区",
      description: "小米董事长，金山软件董事长。业余爱好是天使投资。",
      created_at: "2010-05-31 23:07:59",
      avatar_hd: "",
    });

    const weiboItems = reactive([]);
    const predLabels = reactive([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]);

    function clearData() {
      weiboItems.length = 0;

      for (var i = 0; i < 10; i++) {
        predLabels[i] = -1;
      }

      itemRefs.length = 0;

      weiboUser.nick_name = "";
      weiboUser.followers_count = "";
      weiboUser.friends_count = "";
      weiboUser.location = "";
      weiboUser.description = "";
      weiboUser.created_at = "";
      weiboUser.avatar_hd = "";
      resultId.value = -1;
    }

    const userInfoLoading = ref(null);
    const weiboAnalysisLoading = ref(null);
    // 获取用户信息和近10条微博
    function spiderGetUser() {
      clearData();

      if (input.userId.trim() == "") {
        ElMessage({
          message: "请输入要分析的用户ID",
          type: "warning",
          center: true,
        });
        return;
      }

      const loadingIcon = ElLoading.service({
        // 一定得写loading.value 因为是ref
        target: userInfoLoading.value,
        fullscreen: false,
        text: "",
      });
      axios
        .post("/Api/user_analysis", {
          user_id: input.userId,
          cookie: input.cookie,
        })
        .then(function (response) {
          const data = response.data;

          // 设置微博用户的信息
          weiboUser.nick_name = data.user.nick_name;
          weiboUser.followers_count = data.user.followers_count;
          weiboUser.friends_count = data.user.friends_count;
          weiboUser.location = data.user.location;
          weiboUser.description = data.user.description;
          weiboUser.created_at = data.user.created_at;
          weiboUser.avatar_hd = imgBaseUrl + data.user.avatar_hd;

          // const tweetObj = JSON.parse(data);
          // for (const key in tweetObj) {
          //   if (key.startsWith("tweet")) {
          //     console.log(`${key}: ${jsonObj[key]}`);
          //     weiboItems.push(tweetObj[key]);
          //   }
          // }

          weiboItems.push(data.tweet0);
          weiboItems.push(data.tweet1);
          weiboItems.push(data.tweet2);
          weiboItems.push(data.tweet3);
          weiboItems.push(data.tweet4);
          weiboItems.push(data.tweet5);
          weiboItems.push(data.tweet6);
          weiboItems.push(data.tweet7);
          weiboItems.push(data.tweet8);
          weiboItems.push(data.tweet9);

          // 关闭loading
          loadingIcon.close();
          // Message消息提示
          ElMessage({
            message: "爬虫完成",
            type: "success",
            appendTo: userInfoLoading.value,
            center: true,
          });
        })
        .catch(function (error) {
          console.log(error);
        });
    }

    // 微博内容
    const weiboContent = ref(
      "wow女神又有新片了！！！[送花花][送花花]开机现场的状态也太好了吧，红色卫衣真是红气养人，承包女神的这个笑容了[爱你][抱一抱]算起来已经有快4年没在影院大银幕上看到舒淇了，而且这次还是犯罪悬疑题材，咱就是一整个狠狠期待！希望能早日在影院看到这部《黎明时分见》，迫不及待想看到银幕上的女神舒淇了！"
    );

    const imgUrlAndLabel = reactive([
      {
        imgUrl: require("@/assets/happy.svg"),
        label: "积极",
        style:
          "color:#d81e06; text-align:center; font-weight:bold; margin-top: 40px",
      },
      {
        imgUrl: require("@/assets/mad.svg"),
        label: "愤怒",
        style:
          "color:#d81e06; text-align:center; font-weight:bold; margin-top: 40px",
      },
      {
        imgUrl: require("@/assets/sad.svg"),
        label: "悲伤",
        style:
          "color:#000000; text-align:center; font-weight:bold; margin-top: 40px",
      },
      {
        imgUrl: require("@/assets/fear.svg"),
        label: "恐惧",
        style:
          "color:#000000; text-align:center; font-weight:bold; margin-top: 40px",
      },
      {
        imgUrl: require("@/assets/surprised.svg"),
        label: "惊奇",
        style:
          "color:#75EAE4; text-align:center; font-weight:bold; margin-top: 40px",
      },
      {
        imgUrl: require("@/assets/neutral.svg"),
        label: "无情绪",
        style:
          "color:#000000; text-align:center; font-weight:bold; margin-top: 40px",
      },
    ]);

    const resultId = ref(-1);
    const myPieTable = ref(null);
    // 以下是echart相关
    const mychart = {};
    const pieData = [
      {
        value: 0,
        name: "积极",
      },
      {
        value: 0,
        name: "愤怒",
      },
      {
        value: 0,
        name: "悲伤",
      },
      {
        value: 0,
        name: "恐惧",
      },
      {
        value: 0,
        name: "惊奇",
      },
      {
        value: 0,
        name: "无情绪",
      },
    ];

    let itemRefs = [];
    const setItemRef = (el) => {
      if (el) {
        itemRefs.push(el);
      }
    };
    var wordList = reactive([
      ["你好", 6],
      ["烦人", 12],
    ]);

    function analysisTweets() {
      if (weiboItems.length == 0) {
        ElMessage({
          message: "请先收集微博信息",
          type: "warning",
          center: true,
        });
        return;
      }

      let loadingIcons = [];
      // console.log(weiboItems);
      // console.log(itemRefs);
      for (var i = 0; i < itemRefs.length; i++) {
        loadingIcons.push(
          ElLoading.service({
            // 一定得写loading.value 因为是ref
            target: itemRefs[i],
            fullscreen: false,
            text: "Loading, 请耐心等待约30秒",
          })
        );
      }

      var contents = [];
      for (var i = 0; i < weiboItems.length; i++) {
        contents.push(weiboItems[i].content);
      }

      axios
        .post("/Api/tweets_analysis", {
          num: weiboItems.length,
          contents: contents,
        })
        .then(function (response) {
          const data = response.data;
          // console.log(data);

          for (var i = 0; i < data.labels.length; i++) {
            predLabels[i] = data.labels[i];
          }
          // 关闭loading
          for (var i = 0; i < loadingIcons.length; i++) {
            loadingIcons[i].close();
          }

          // Message消息提示
          ElMessage({
            message: "分析完成",
            type: "success",
            // appendTo: userInfoLoading.value,
            center: true,
          });

          // 生成echarts
          // 清空pieData的数据
          for (var i = 0; i < 6; i++) {
            pieData[i].value = 0;
          }
          // 先统计各情绪的数量
          var maxNum = 0;
          var maxIndex = 0;
          for (var i = 0; i < weiboItems.length; i++) {
            pieData[predLabels[i]].value++;
            if (pieData[predLabels[i]].value > maxNum) {
              maxNum = pieData[predLabels[i]].value;
              maxIndex = predLabels[i];
              // console.log("maxIndex = " + maxIndex);
            }
          }

          // 显示饼图
          resultId.value = maxIndex;
          drawPieTable();

          //词云

          wordCloudUrl.value = data.imgUrl;

          // console.log(wordCloudUrl);
        })
        .catch(function (error) {
          console.log(error);
        });
    }

    const pieName = [];
    const myChartStyle = { float: "left", width: "100%", height: "300px" }; //图表样式

    function drawPieTable() {
      for (let i = 0; i < pieData.length; i++) {
        pieName[i] = pieData[i].name;
      }
      const option = {
        legend: {
          // 图例
          data: pieName,
          right: "0%",
          top: "30%",
          orient: "vertical",
        },
        title: {
          // 设置饼图标题，位置设为顶部居中
          text: "各类情绪占比",
          top: "0%",
          left: "center",
        },
        series: [
          {
            type: "pie",
            label: {
              show: true,
              formatter: "{b} : {c} ({d}%)", // b代表名称，c代表对应值，d代表百分比
            },
            left: "0%",
            radius: "30%", //饼图半径
            data: pieData,
          },
        ],
      };
      let newPromise = new Promise((resolve) => {
        resolve();
      });
      var myChart;
      // 异步执行echarts的初始化函数
      newPromise.then(() => {
        myChart = echarts.init(document.getElementById("myPieTable"));
        myChart.setOption(option);
        //随着屏幕大小调节图表
        window.addEventListener("resize", () => {
          myChart.resize();
        });
      });
    }





    return {
      input,
      urlTemp,
      wordCloudUrl,
      spiderGetUser,
      itemRefs,
      setItemRef,
      weiboContent,
      analysisTweets,
      userInfoLoading,
      imgUrlAndLabel,
      resultId,
      weiboItems,
      weiboUser,
      pieName,
      myChartStyle,
      predLabels,
      weiboAnalysisLoading,
      myPieTable,
      wordList,
    };
  },
};
</script>

<style scoped>
.wrapper {
  width: 90%;
  margin: 15px 5%;
}

.tweetWrapper {
  width: 90%;
  margin: 15px 5%;
}

.shadow {
  box-sizing: border-box;
  box-shadow: var(--el-box-shadow-light);
  width: 100%;
  padding: 10px;
  text-align: left;
}

.margin-bottom-0 .el-form-item {
  margin-bottom: 0 !important;
}

#user-info .el-row {
  padding-left: 25px;
}

#nickname {
  font-size: 20px;
  height: 30px;
  font-weight: bold;
}

#nickname-small {
  font-size: 14px;
  height: 25px;
  font-weight: bold;
}

#desc {
  font-size: 12px;
  color: #939393;
}

#result {
  margin: 10px 0 0 0;
}


</style>
