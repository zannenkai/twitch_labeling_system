
// TwitchチャットからのメッセージをPythonに送信するためのJavaScriptコード
// PythonのWebSocketサーバーのURL
const serverUrl = 'ws://localhost:8000/';

// WebSocket接続の作成
const socket = new WebSocket(serverUrl);

// WebSocketが接続されたときの処理
socket.onopen = () => {
};

// Twitchチャットからのメッセージを受信したときの処理
function handleMessage(event) {
    const json = JSON.stringify(event);
    // メッセージをPythonに送信
    socket.send(json);
}
// websocket受信
socket.addEventListener('message', function (event) {
    const message = JSON.parse(event.data);
    // メッセージを処理するための任意の処理をここに記述
    const chatId = message["chatId"]
    const label = message["predictions"][0]["label"]
    const predictions = message["predictions"]
    applyLabel(chatId, label, predictions)
});

const applyLabel = (chatId, label, predictions) => {
    if (label == "その他") {
        return 0
    }
    targetEle = document.querySelector("#chatId-" + chatId)
    const commentEle = targetEle.querySelector('[data-a-target="chat-line-message-body"]')
    commentEle.setAttribute('label', label)
    commentEle.classList.add("hoverable")

    // ツールチップ要素の作成
    const predictionsDiv = document.createElement('div');
    predictionsDiv.className = 'predictions';
    commentEle.appendChild(predictionsDiv);
    for (const p of predictions) {
        const p_label = p["label"];
        const p_rate = p["rate"];
    
        const predictionDiv = document.createElement('div');
        predictionDiv.textContent = `${p_label}: ${p_rate}`;
    
        // predictionsDiv に新しい div を追加
        predictionsDiv.appendChild(predictionDiv);
    }

}
const getChatData = (ele) => {
    const messageEle = ele.querySelector(".Layout-sc-1xcs6mc-0.chat-line__no-background")
    if(messageEle == null){
        return null;
    }
    let text = ""
    const author = messageEle.querySelector(".chat-author__display-name").innerText;
    const textEle = messageEle.querySelectorAll('[data-a-target="chat-line-message-body"] .text-fragment')
    // emote用の処理　textからemoteを削除する
    textEle.forEach((e) => {
        text += e.innerText
    });

    return { author: author, text: text }
}

const isTarget = (author, text) => {
    // 除外するチャット
    const ignoredKeyword = ["subscribed with Prime", "subscribed at Tier"];
    // 除外するリンク
    const ignoredLink = [
        "https://www.twitch.tv/[a-zA-Z0-9_]+/clip/",
        "https://clips.twitch.tv/"
    ];
    // 除外する送信者
    const ignoredSender = "Nightbot";

    if (
        author === ignoredSender ||
        ignoredKeyword.some(keyword => text.includes(keyword)) ||
        ignoredLink.some(pattern => new RegExp(pattern).test(text) ||
            text == "")
    ) {
        return false
    } else {
        return true
    }
}
// Twitchチャットの監視を開始
function startChatMonitoring() {
    let chatId = 1;
    const wrapperElement = document.querySelector(".chat-scrollable-area__message-container"); // Twitchチャットの要素
    // Twitchチャットの変更を監視し、メッセージが追加されたときにhandleMessageを呼び出す
    const observer = new MutationObserver(mutations => {
        mutations.forEach(mutation => {
            // 新しいメッセージが追加されたら、handleMessageを呼び出す
            if (mutation.addedNodes.length > 0) {
                // 新しいメッセージが最新のチャットなら
                data = getChatData(mutation.addedNodes[0])

                // コメントが検出対象だったらモデルに通す
                if (isTarget(data.author, data.text)) {
                    mutation.addedNodes[0].id = "chatId-" + chatId;
                    handleMessage({ data: data.text, chatId: chatId });
                    chatId++;
                }
            }
        });
    });
    // Twitchチャット要素の変更を監視する設定
    observer.observe(wrapperElement, { childList: true });
}
// DOM読み込まれたらTwitchチャットの監視を開始
window.addEventListener('load', startChatMonitoring);
