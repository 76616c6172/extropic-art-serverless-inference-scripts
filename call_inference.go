package main

import (
	"fmt"
	"log"
	"os/exec"
)

func main() {
	MODEL := "0"
	PROMPT := "Portrait of a cute girl, riven, league of legends character art, Artstation, WLOP"
	WIDTH := "512"
	HEIGHT := "768"
	STEPS := "75"
	SCALE := "7"
	SEED := "1432518515"

	cmd := exec.Command("./run_inference.py", MODEL, PROMPT, SEED, WIDTH, HEIGHT, STEPS, SCALE)
	out, err := cmd.Output()
	if err != nil {
		log.Println(err)
		return
	}
	fmt.Println(string(out))
}
