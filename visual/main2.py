from nicegui import ui, context


def locked_page_height():
    """
    锁定页面内容区域的高度，等于屏幕高度减去顶部和底部的高度
    意味着 内容区创建的第一个容器，可以通过 h-full 让其高度等于屏幕高度(减去顶部和底部的高度).

    此函数创建时的 nicegui 版本为:1.4.20
    """
    client = context.get_client()
    q_page = client.page_container.default_slot.children[0]
    q_page.props(
        ''':style-fn="(offset, height) => ( { height: offset ? `calc(100vh - ${offset}px)` : '100vh' })"'''
    )
    client.content.classes("h-full")


locked_page_height()

with ui.header().classes():
    ui.label("这是标题")


template = """
    'left-drawer content'
    'left-drawer bottom-bar'
"""


with ui.grid(rows="1fr auto", columns="auto 1fr").classes(
    "w-full h-full overflow-hidden gap-y-4"
).style(f"grid-template-areas: {template};"):
    # 左侧抽屉
    with ui.column(wrap=False).classes("p-4 overflow-y-auto").style(
        "grid-area:left-drawer"
    ):
        for i in range(100):
            ui.label("这是page_sticky左抽屉")

    # 右侧内容
    with ui.column().classes("p-4 overflow-y-auto").style("grid-area:content"):
        for i in range(100):
            ui.label("内容区域")

    # 底部栏
    with ui.column().classes(" items-center").style("grid-area:bottom-bar"):
        #
        with ui.row().classes("items-center"):
            ui.textarea(placeholder="message chart gpt...").classes(
                "min-w-[50vw]"
            ).props("desen outlined autogrow")

            ui.button("附件").props("flat")

        #
        ui.label(
            "ChatGPT can make mistakes. Consider checking important information. Read our Terms and Privacy Policy."
        )
ui.run()