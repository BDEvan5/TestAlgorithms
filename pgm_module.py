
import pgm_reader

f = 'maps/race_track.pgm'
reader = pgm_reader.Reader()
image = reader.read_pgm(f)
reader.show_img()
