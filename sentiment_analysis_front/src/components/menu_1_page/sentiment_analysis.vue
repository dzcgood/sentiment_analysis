<template>
  <div>
    <!--输入部分-->
    <div class="input" ref="loading">
      <div class="wrapper">
        <el-row id="info" align="middle">
          <el-col :span="6">
            <span>请输入一段需要分析的文本：</span>
          </el-col>

          <el-col :span="2">
            <el-button type="primary" text @click="changeInput"
              >换一个示例</el-button
            >
          </el-col>
        </el-row>

        <el-row id="textarea">
          <el-col :span="24">
            <el-input
              type="textarea"
              v-model="input"
              :placeholder="placeholder[curIndex]"
              rows="4"
              maxlength="140"
              show-word-limit
            />
          </el-col>
        </el-row>
        <el-row align="middle" justify="space-between" id="do_analysis">
          <el-col :span="4" style="color: #c8c9cc">
            <span>最多输入</span>
            <span style="color: #eebe77">140</span>
            <span>字</span>
          </el-col>

          <el-col :span="2">
            <el-button type="primary" @click="analysis">开始分析</el-button>
          </el-col>
        </el-row>
        <el-row align="middle" id="result" v-if="resultId != -1">
          <el-col :span="3">情感分析结果</el-col>
          <el-col :span="1">
            <el-image
              style="width: 50px; height: 50px"
              :src="imgUrlAndLabel[resultId].imgUrl"
              fit
            />
          </el-col>
          <el-col :span="2" :style="imgUrlAndLabel[resultId].style">{{
            imgUrlAndLabel[resultId].label
          }}</el-col>
        </el-row>
      </div>
    </div>

    <!--历史记录部分-->
    <div id="history">
      <div class="input">
        <div class="wrapper">
          <el-row>
            <el-col :span="12">
              <span>历史分析：</span>
            </el-col>
            <el-col :span="12" v-if="tableData.data.length == 0">
              <span>无相关记录</span>
            </el-col>
          </el-row>
        </div>
      </div>

      <div class="input" v-if="tableData.data.length > 0">
        <div class="wrapper">
          <el-row>
            <el-table
              :data="tableData.data"
              style="width: 100%"
              height="450"
              tableLayout="fixed"
              stripe
            >
              <el-table-column fixed prop="date" label="Date" />
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
    // 预设的文本
    const placeholder = reactive([
      "今天天气真好，心情舒畅！",
      "一出门摔了一跤，真是服了。",
      "听故事的时候,你总喜欢眼巴巴的问,后来呢,可是当后来,自己成为讲故事的人,才发现,后来故事和话语就在嘴边,后来眼泪几度泪凝于睫,是真的说不下去了。。",
      "原来介玩意是叫草子糕啊!我活了30多年一直以为是槽子糕了!涨姿势了!",
    ]);
    const curIndex = ref(0);
    const input = ref("");

    // 预设之间切换文本
    function changeInput() {
      curIndex.value = (curIndex.value + 1) % placeholder.length;
      input.value = "";
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

    const resultId = ref(-1);

    // 需要显示loading的dom元素
    const loading = ref(null);
    
    // 分析文本情感
    function analysis() {
      if (input.value == "") {
        input.value = placeholder[curIndex.value];
      }
      console.log(input.value);
      const loadingIcon = ElLoading.service({
        // 一定得写loading.value 因为是ref
        target: loading.value,
        fullscreen: false,
        text: "Loading",
      });
      axios
        .post("/Api/sentiment_analysis", {
          content: input.value,
        })
        .then(function (response) {
          console.log(response);
          resultId.value = response.data["pred"];
          // 关闭loading
          loadingIcon.close();
          // Message消息提示
          ElMessage({
            message: "分析完成",
            type: "success",
            appendTo: loading.value,
            center: true,
          });
          // 添加历史记录
          setHistory(input.value, imgUrlAndLabel[resultId.value].label);
        })
        .catch(function (error) {
          console.log(error);
        });
    }

    // 历史记录
    const tableData = reactive({
      data: [
        {
          date: "2016-05-03",
          content: "今天天气真好，心情舒畅!",
          result: "sad",
        },
        {
          date: "2016-05-03",
          content: "今天天气真好，心情舒畅！",
          result: "happy",
        },
        {
          date: "2016-05-03",
          content: "今天天气真好，心情舒畅！",
          result: "happy",
        },
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
    function setHistory(content, label) {
      console.log(tableData.data);
      if (tableData.data.length >= 20) {
        tableData.data.pop();
      }
      tableData.data.unshift({
        date: getNowDate(),
        content: content,
        result: label,
      });
      localStorage.setItem(
        "sentimentAnalysisHistory",
        JSON.stringify(tableData.data)
      );
    }

    // 移除一条历史记录
    function removeOneHistory(index) {
      console.log(index);
      tableData.data.splice(index, 1);
      localStorage.setItem(
        "sentimentAnalysisHistory",
        JSON.stringify(tableData.data)
      );
      ElNotification({
        title: "Success",
        message: "成功删除一条记录",
        type: "success",
      });
    }

    // 加载时获取历史记录
    onMounted(() => {
      var history = JSON.parse(
        localStorage.getItem("sentimentAnalysisHistory")
      );
      if (history != null) {
        tableData.data = history;
      }
    });

    // RETURN
    return {
      input,
      placeholder,
      curIndex,
      changeInput,
      analysis,
      tableData,
      imgUrlAndLabel,
      resultId,
      loading,
      removeOneHistory,
      setHistory,
    };
  },
};
</script>

<style scoped>
.input {
  width: 90%;
  margin: 15px 5%;
}
#info {
  text-align: left;
  margin-bottom: 5px;
}

.wrapper {
  box-sizing: border-box;
  box-shadow: var(--el-box-shadow-light);
  width: 100%;
  padding: 10px;
  text-align: left;
}

#textarea {
  margin-bottom: 10px;
}

#do_analysis {
  border-bottom: 1px solid var(--el-border-color);
  padding-bottom: 10px;
}

#result {
  margin: 10px 0 0 0;
}

#history {
  width: 100%;
  border-top: 1px solid var(--el-border-color);
  padding-top: 10px;
}
</style>
