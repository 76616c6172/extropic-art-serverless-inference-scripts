{
	description = "defines the nix environment the workers depend on";
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
              pkgs.Python311
              pkgs.python3Packages.modal-client
						];
          };
        });
    };
}
