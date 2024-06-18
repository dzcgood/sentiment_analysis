// 路由，页面切换
import { createRouter, createWebHashHistory } from "vue-router";

const routes = [
    // 每个对象就是一个界面
    // 登录
    {
        path: '/', 
        name: 'index', 
        component:()=>import(/*webpackChunkName:'index'*/'@/components/index/index.vue'),
        redirect: '/sentiment_analysis',
        // 二级路由
        children: [
            {
                // 文本情感分析
                path: '/sentiment_analysis',
                name: 'sentiment_analysis',
                component:()=>import(/*webpackChunkName:'sentiment_analysis'*/'@/components/menu_1_page/sentiment_analysis.vue')
            },
            {
                // 微博情感分析
                path: '/analysis_by_weibo_id',
                name: 'analysis_by_weibo_id',
                component:()=>import(/*webpackChunkName:'sentiment_analysis'*/'@/components/menu_2_page/analysis_by_weibo_id.vue')
            },
            {
                // 微博用户情感分析
                path: '/analysis_weibo_user',
                name: 'analysis_weibo_user',
                component:()=>import(/*webpackChunkName:'sentiment_analysis'*/'@/components/menu_3_page/analysis_weibo_user.vue')
            },
        ]
    }
]

const router = createRouter({
    history: createWebHashHistory(),
    routes
})

export default router

