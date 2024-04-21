from __future__ import annotations

from typing import Callable, Optional, Protocol

from nicegui import ui


class Item(Protocol):
    title: str


dragged: Optional[card] = None


class lazy_drop(ui.column):
    def __init__(
            self, name: str, on_drop: Optional[Callable[[Item, str], None]] = None
    ) -> None:
        super().__init__()
        with self.classes("bg-blue-grey-2 min-w-60 p-4 rounded shadow-2"):
            ui.label(name).classes("text-bold ml-1")
            # 用来放置卡片
            self.cards_container = ui.row(wrap=False)

        self.name = name
        self.on("dragover.prevent", self.highlight)
        self.on("dragleave", self.unhighlight)
        self.on("drop", self.move_card)
        self.on_drop = on_drop

    def __enter__(self):
        if not hasattr(self, "cards_container"):
            return super().__enter__()
        self.cards_container.__enter__()
        return self

    def __exit__(self, *args):
        if not hasattr(self, "cards_container"):
            return super().__exit__(*args)
        self.cards_container.__exit__(*args)

    def highlight(self) -> None:
        self.classes(remove="bg-blue-grey-2", add="bg-blue-grey-3")

    def unhighlight(self) -> None:
        self.classes(remove="bg-blue-grey-3", add="bg-blue-grey-2")

    def move_all_cards(self, descendant: lazy_drop) -> None:
        for card in list(self.cards_container):
            card.move(descendant.cards_container)

    def move_card(self) -> None:
        global dragged
        self.unhighlight()
        dragged.parent_slot.parent.remove(dragged)
        with self.cards_container:
            card(dragged.item)
        self.on_drop(dragged.item, self.name)
        dragged = None


class card(ui.card):
    def __init__(self, item: Item) -> None:
        super().__init__()
        self.item = item
        with self.props("draggable").classes("w-full cursor-pointer bg-grey-1"):
            ui.label(item.title)
        self.on("dragstart", self.handle_dragstart)

    def handle_dragstart(self) -> None:
        global dragged
        dragged = self
