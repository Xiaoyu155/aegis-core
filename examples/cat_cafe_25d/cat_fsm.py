class CatFSM:
    def update(self, cat, dt):
        if dt < 0: raise ValueError('negative dt')
        if cat.state == 'idle' and dt > 0:
            cat.state='walking'
        elif cat.state == 'walking':
            cat.x += dt * 2
            if cat.x >= 10: cat.state='resting'
        elif cat.state == 'resting':
            cat.state='idle'
