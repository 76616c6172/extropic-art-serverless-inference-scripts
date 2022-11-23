{
	description = "defines dependencies in a nix flake";
	inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
	outputs = { self, nixpkgs }:
		let
      supportedSystems = [ "x86_64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      nixpkgsFor = forAllSystems (system: import nixpkgs { inherit system; });
		in
    {
      packages = forAllSystems (system:
        let
          pkgs = nixpkgsFor.${system};
        in
        {
          default = pkgs.mkShell {
						packages = [
              pkgs.python311.buildPythonPackage {
                name = "whisperlib";
                version = "0.0.1";
                # src = ./app;
                propagatedBuildInputs = [
                  pkgs.python3Packages.openai-whisper 
                  pkgs.python3Packages.fastapi
                ];
            };
						];
          };
        };
        );
    };
}
