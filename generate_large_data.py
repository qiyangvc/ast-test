"""生成大数据集脚本"""
import os
import random

# 正常短信模板
normal_templates = [
    '今天天气不错，适合出去走走',
    '明天一起吃饭吧',
    '周末有空吗？一起看电影',
    '会议改到下午三点了',
    '记得带伞，今天有雨',
    '生日快乐！祝你天天开心',
    '下班后来我办公室一趟',
    '快递已经到了，记得签收',
    '明天放假，不用上班',
    '最近工作怎么样？',
    '周末一起去爬山吧',
    '帮我带一份午饭，谢谢',
    '晚上聚餐，六点准时到',
    '这本书很好看，推荐给你',
    '明天早上八点开会',
    '家里的水电费该交了',
    '孩子放学了，去接一下',
    '今天加班，晚点回家',
    '周末想去哪里玩？',
    '最近身体还好吗？'
]

# 垃圾短信模板
spam_templates = [
    '免费领取手机话费充值卡，立即点击领取',
    '恭喜您中了大奖，请点击链接领取',
    '您的账户有异常，请立即登录验证',
    '低价出售手机，正品保证，货到付款',
    '贷款急速审批，当天放款，无需抵押',
    '代办各种证件，快速办理，绝对保密',
    '投资股票，稳赚不赔，日入千金',
    '刷单返利，轻松赚钱，日结工资',
    '您有一份快递未签收，请点击链接查看',
    '中奖通知：您已获得苹果手机一部',
    '减肥产品，无效退款，月瘦十斤',
    '增高药，三个月增高五厘米',
    '学历提升，快速拿证，国家承认',
    '信用卡代办，额度高，下卡快',
    '赌场在线，24小时营业，首充送豪礼',
    '色情视频，免费观看，更新不断',
    '开发票，正规发票，点数优惠',
    '走私奢侈品，低价出售，质量保证',
    '黑客服务，盗号追款，无所不能',
    '代考代练，轻松过关，价格实惠'
]

# 生成大数据集
def generate_large_dataset(output_dir, num_normal=50000, num_spam=50000):
    """生成大规模数据集"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成正常短信
    normal_file = os.path.join(output_dir, 'msgpass.log.seg')
    with open(normal_file, 'w', encoding='utf-8') as f:
        for i in range(num_normal):
            template = random.choice(normal_templates)
            # 添加一些随机变化
            if random.random() > 0.7:
                template = template.replace('明天', '后天') if '明天' in template else template
            if random.random() > 0.8:
                template = template.replace('今天', '昨天') if '今天' in template else template
            f.write(template + '\n')
    
    # 生成垃圾短信
    spam_file = os.path.join(output_dir, 'msgspam.log.seg')
    with open(spam_file, 'w', encoding='utf-8') as f:
        for i in range(num_spam):
            template = random.choice(spam_templates)
            # 添加一些随机变化
            if random.random() > 0.7:
                template = template.replace('点击', '访问') if '点击' in template else template
            if random.random() > 0.8:
                template = template.replace('免费', '限时') if '免费' in template else template
            f.write(template + '\n')
    
    print(f"生成完成！")
    print(f"  正常短信: {num_normal} 条")
    print(f"  垃圾短信: {num_spam} 条")
    print(f"  总数据量: {num_normal + num_spam} 条")

if __name__ == '__main__':
    generate_large_dataset('data/msglog', num_normal=50000, num_spam=50000)