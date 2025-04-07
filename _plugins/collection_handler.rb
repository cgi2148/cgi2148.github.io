module Jekyll
  class CollectionDirectoryReader < Generator
    safe true

    def generate(site)
      # 외부 포스트 디렉토리 탐색
      external_post_dirs = ['konkuk/rag', 'konkuk/rag-code', 'konkuk/mcp', 'mustree']
      
      external_post_dirs.each do |dir|
        if Dir.exist?(dir)
          Dir.glob(File.join(dir, "**/*.{markdown,md}")).each do |file|
            # 날짜 형식을 포함하는 파일 이름 확인 (예: 2025-04-07-*.md)
            if File.basename(file) =~ /^\d{4}-\d{2}-\d{2}/
              # 파일을 읽고 처리
              read_file(site, file)
            end
          end
        end
      end
    end

    def read_file(site, file)
      # 파일 내용 읽기
      content = File.read(file)
      
      # YAML 프론트매터를 가지고 있는지 확인
      if content =~ /\A---\s*\n(.*?)\n---\s*\n/m
        data = YAML.safe_load($1)
        data['path'] = file # 원본 파일 경로 저장
        
        # 문서 생성
        doc = Jekyll::Document.new(
          file, 
          { site: site, collection: site.posts }
        )
        
        # 포스트에 데이터 추가
        site.posts.docs << doc
      end
    end
  end
end 