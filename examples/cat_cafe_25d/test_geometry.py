from entities import Cat
from depth_sorter import DepthSorter
from render_loop import RenderLoop

def test_depth():
    cats=[Cat('back',10,99),Cat('front',20,1),Cat('overlay',0,0,1)]
    assert [c.ident for c in DepthSorter().order(cats)]==['back','front','overlay']

def test_memory():
    loop=RenderLoop([Cat('a',1,1)])
    buffer=loop.draw_buffer
    for _ in range(1000): loop.tick(.01)
    assert loop.draw_buffer is buffer
    assert len(loop.history)<=1
